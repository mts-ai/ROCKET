import torch
import torch.nn as nn
import gc
from swiftsvd.utils.model_utils import get_weight_transposed, get_k_and_sparsity, frobenius_distance
from swiftsvd.calib.calib import Calib

def quantize_int8_per_row(x: torch.Tensor):
    """Quantize matrix per row (dim=0). x: (M, N) → scales: (M,)"""
    assert x.dim() == 2, "Only 2D tensors supported"
    eps = 1e-8
    max_val = x.abs().amax(dim=1, keepdim=True).clamp(min=eps)  # (M, 1)
    scale = max_val / 127.0
    x_int8 = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return x_int8, scale.squeeze(1)  # scales: (M,)


def dequantize_int8_per_row(x_int8: torch.Tensor, scales: torch.Tensor):
    """Dequantize per-row quantized matrix."""
    assert x_int8.dim() == 2 and scales.dim() == 1
    assert x_int8.shape[0] == scales.shape[0]
    return x_int8.float() * scales.unsqueeze(1)  # (M, N) * (M, 1)
    
def svd_with_magnitude_sparsity_on_v(
    w1,
    name,
    index,
    cr_target,
    ks_ratio,
    calib_data,
    dobi_like = False,
    down=False
):
    if not isinstance(index, list):
        index = [index]

    ss, inv_s = Calib.get_s_inv_s(index, name, "llama3", calib_data)
    ss = ss.double().cuda()
    inv_s = inv_s.double().cuda()
    w1 = w1.double().cuda()

    with torch.no_grad():
        x = ss @ w1
        u, sigma, v = torch.linalg.svd(x, full_matrices=False)
        d1, d2 = w1.shape
        r, sparsity_ratio = get_k_and_sparsity(cr_target, d1, d2, 1, ks_ratio)
        r = max(1, min(r, min(d1, d2)))

        u = u[:, :r]
        sigma = sigma[:r]
        v = v[:r, :]
        sqrt_sigma = sigma ** 0.5
        u = u @ torch.diag(sqrt_sigma)
        v = torch.diag(sqrt_sigma) @ v

        sparsity_ratio = min(max(sparsity_ratio, 0.0), 1.0)
        if sparsity_ratio <= 0.0:
            v_sparse = v.clone()
        elif sparsity_ratio >= 1.0:
            v_sparse = torch.zeros_like(v)
        else:
            G_x = ss @ ss.t()
            G_y = w1.t() @ G_x @ w1
            output_importance = torch.sqrt(torch.diag(G_y) + 1e-8)
            v_weighted = v * output_importance.unsqueeze(0)
            v_abs = v_weighted.abs()
            num_rows = v_abs.shape[0]
            num_keep = max(1, int(num_rows * (1 - sparsity_ratio)))
            kth = num_rows - num_keep
            if kth > 0:
                thresholds = torch.kthvalue(v_abs, kth, dim=0).values
                mask = v_abs > thresholds.unsqueeze(0)
            else:
                mask = torch.ones_like(v_abs, dtype=torch.bool)
            v_sparse = v * mask

        G = v_sparse @ v_sparse.t()
        trace_G = torch.trace(G)
        reg_eps = max(1e-6, 1e-3 * (trace_G / r).item()) if r > 0 else 1e-6
        G_reg = G + torch.eye(r, device=G.device, dtype=G.dtype) * reg_eps
        WVt = x @ v_sparse.t()
        try:
            cholesky_factor = torch.linalg.cholesky(G_reg)
            U_opt = torch.cholesky_solve(WVt.t().contiguous(), cholesky_factor).t()
        except RuntimeError:
            U_opt = torch.linalg.solve(G_reg, WVt.t()).t()

        u_final = inv_s @ U_opt
        v_final = v_sparse
        if dobi_like:
            u_int8, u_scales = quantize_int8_per_row(u_final)   # u_out: (4096, r)
            v_int8, v_scales = quantize_int8_per_row(v_final)   # v_out: (r, d2)
            u_final = dequantize_int8_per_row(u_int8, u_scales)
            v_final = dequantize_int8_per_row(v_int8, v_scales)
        w1_recon = u_final @ v_final
        err_v = frobenius_distance(w1_recon, w1)
        print(f"[{name}][{index}] CR={cr_target:.3f} | Rank={r} | Sparsity={sparsity_ratio:.3f} | Err={err_v:.6f}")
   
        
    del sigma, w1, ss, inv_s, u, v, x, G, G_reg, WVt
    gc.collect()
    torch.cuda.empty_cache()
    return u_final.cpu(), v_final.cpu()