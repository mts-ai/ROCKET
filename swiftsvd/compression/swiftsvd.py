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
    cr_ks,
    calib_data,
    dobi_like = False,
    down=False,
    adam_refine_steps=0
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
        _, importance,_ = torch.linalg.svd(w1, full_matrices=False)
        d1, d2 = w1.shape
        r, sparsity_ratio = get_k_and_sparsity(cr_ks['cr'], d1, d2, 1, cr_ks['ks'])
        r = max(1, min(r, min(d1, d2)))

        # After SVD
        u = u[:, :r]          # (d1, r)
        sigma = sigma[:r]     # (r,)
        v = v[:r, :]          # (r, d2)
        v_full = torch.diag(sigma) @ v  # (r, d2)
        
        # 1. Whitened-space importance (already in v_full)
        imp_x = v_full.abs()  # (r, d2)
        
        # 2. Original-space importance
        A = inv_s @ u         # (d1, r) — how each latent dim maps to w1 input dims
        latent_norms = torch.norm(A, dim=0, p=2)  # (r,) — ||A[:,i]||_2
        imp_w = v_full.abs() * latent_norms.unsqueeze(1)  # (r, d2)
        
        # 3. Combine (tunable lambda)
        lambda_reg = 0.5  # try 0.1, 0.3, 0.5 — start small
        #imp_combined = (1 - lambda_reg) * imp_x + lambda_reg * imp_w
        imp_combined = torch.exp(
            (1 - lambda_reg) * torch.log(imp_x + 1e-12) +
            lambda_reg * torch.log(imp_w + 1e-12)
        )
        # 4. Per-column sparsification on imp_combined
        sparsity_ratio = min(max(sparsity_ratio, 0.0), 1.0) + 0.03
        if sparsity_ratio <= 0.0:
            v_sparse = v_full.clone()
        elif sparsity_ratio >= 1.0:
            v_sparse = torch.zeros_like(v_full)
        else:
            v_abs = imp_combined
            num_rows = v_abs.shape[0]
            num_keep_per_col = max(1, int(num_rows * (1 - sparsity_ratio)))
            kth = num_rows - num_keep_per_col
        
            if kth > 0:
                thresholds = torch.kthvalue(v_abs, kth, dim=0).values  # (d2,)
                mask = v_abs >= thresholds.unsqueeze(0)
            else:
                mask = torch.ones_like(v_abs, dtype=torch.bool)
        
            v_sparse = v_full * mask  # ← apply to original v_full
        # After initial v_sparse
        error_per_entry = (v_full * (~mask)).abs()  # contribution to error if zeroed
        # But since u is orthonormal, error ∝ |v_full|^2
        recovery_score = error_per_entry
        
        # Recover top 1% of zeros (or fixed number)
        num_recover = max(1, int(mask.numel() * 0.03))
        flat_score = recovery_score.view(-1)
        if num_recover < flat_score.numel():
            threshold = torch.kthvalue(flat_score, flat_score.numel() - num_recover).values
            recover_mask = (~mask) & (recovery_score >= threshold)
            mask = mask | recover_mask
            v_sparse = v_full * mask
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

    if adam_refine_steps > 0:
        # Initialize trainable parameters
        U_param = nn.Parameter(U_opt.clone())  # (m, r)
        V_param = nn.Parameter(v_sparse.clone())  # (r, n)

        # Fix sparsity: register mask as buffer
        mask_v = mask.to(V_param.device)

        optimizer = torch.optim.Adam([U_param, V_param], lr=1e-3, amsgrad=False)
        target = x.detach()  # = ss @ w1, whitened target

        for step in range(adam_refine_steps):
            optimizer.zero_grad()
            # Apply sparsity mask
            V_masked = V_param * mask_v
            recon = U_param @ V_masked
            diff = recon - target
            loss_fro_sq = torch.norm(diff, p='fro') ** 2
            target_norm_sq = torch.norm(target, p='fro') ** 2 + 1e-12  # avoid division by zero
            
            loss = loss_fro_sq / target_norm_sq  # ✅ Relative MSE (squared)
            loss.backward()
            optimizer.step()

            # Optional: log every 20 steps
            if step % 20 == 0:
                print(f"  Adam step {step}, loss: {loss.item():.6f}")

        # Final masked V
        with torch.no_grad():
            V_final_sparse = V_param * mask_v
            U_final_tilde = U_param
    else:
        U_final_tilde = u
        V_final_sparse = v_sparse
    
        # --- Map back to original space ---
    with torch.no_grad():
        u_final = inv_s @ U_final_tilde
        v_final = V_final_sparse
        u_final2 = inv_s @ U_opt
        v_final2 = v_sparse
        # Correct error: compare to original w1
        w1_recon = u_final @ v_final
        w2_recon = u_final2 @ v_final2
        err_v = frobenius_distance(w1_recon, w1)
        err_v2 = frobenius_distance(w2_recon, w1)
        print(f"Final reconstruction error after Adam refinement: {err_v:.6f}")
        if err_v2 < err_v:
            u_final = u_final2
            v_final = v_final2
        if dobi_like:
            u_int8, u_scales = quantize_int8_per_row(u_final)   # u_out: (4096, r)
            v_int8, v_scales = quantize_int8_per_row(v_final)   # v_out: (r, d2)
            u_final = dequantize_int8_per_row(u_int8, u_scales)
            v_final = dequantize_int8_per_row(v_int8, v_scales)
        w1_recon = u_final @ v_final
        err_v = frobenius_distance(w1_recon, w1)
        print(f"[{name}][{index}] CR, KS={cr_ks} | Rank={r} | Sparsity={sparsity_ratio:.3f} | Err={err_v:.6f}")
   
        
    del sigma, w1, ss, inv_s, u, v, x, G, G_reg, WVt
    gc.collect()
    torch.cuda.empty_cache()
    return u_final.cpu(), v_final.cpu()
