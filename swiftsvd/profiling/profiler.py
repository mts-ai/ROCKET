import torch
import numpy as np
from tqdm import tqdm
from swiftsvd.utils.model_utils import get_weight_transposed, get_k_and_sparsity, frobenius_distance
from swiftsvd.calib.calib import Calib

def profile_layer_once(w1, name, index, calib_data, cr_candidates, L=1, ks_ratio=2.0):
    if not isinstance(index, list):
        index = [index]
    d1, d2 = w1.shape
    orig_params = d1 * d2

    ss, inv_s = Calib.get_s_inv_s(index, name, "llama3", calib_data)
    ss = ss.double().cuda()
    inv_s = inv_s.double().cuda()
    w1 = w1.double().cuda()

    with torch.no_grad():
        x = ss @ w1
        u_full, sigma_full, v_full = torch.linalg.svd(x, full_matrices=False)
        max_r = min(d1, d2)
        sigma_full = sigma_full[:max_r]
        u_full = u_full[:, :max_r]
        v_full = v_full[:max_r, :]

        G_x = ss @ ss.t()
        G_y = w1.t() @ G_x @ w1
        output_importance = torch.sqrt(torch.diag(G_y) + 1e-8)

        results = []
        for cr in cr_candidates:
            r_float, sparsity_ratio = get_k_and_sparsity(cr, d1, d2, L, ks_ratio)
            r = max(1, min(int(r_float), max_r))

            u = u_full[:, :r]
            sigma = sigma_full[:r]
            v = v_full[:r, :]
            sqrt_sigma = sigma ** 0.5
            u = u @ torch.diag(sqrt_sigma)
            v = torch.diag(sqrt_sigma) @ v

            if 0 < sparsity_ratio < 1:
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
            else:
                v_sparse = v

            G = v_sparse @ v_sparse.t()
            G_reg = G + torch.eye(r, device=G.device, dtype=G.dtype) * 1e-8
            WVt = x @ v_sparse.t()
            try:
                cf = torch.linalg.cholesky(G_reg)
                U_opt = torch.cholesky_solve(WVt.t().contiguous(), cf).t()
            except:
                U_opt = torch.linalg.solve(G_reg, WVt.t()).t()

            w1_recon = (inv_s @ U_opt) @ v_sparse
            err = frobenius_distance(w1_recon, w1).item()
            params_used = r * (d1 + d2)
            actual_cr = params_used / orig_params
            results.append((cr, actual_cr, err))

    del ss, inv_s, w1, x, u_full, sigma_full, v_full
    torch.cuda.empty_cache()
    return results

def profile_all_layers(model, module_names, calib_data, cr_candidates, ks_ratio=2.0):
    layer_profiles = []
    total_params = 0
    for name in module_names:
        for idx in tqdm(range(len(model.model.layers)), desc=f"Profiling {name}"):
            w1 = get_weight_transposed(model, name, idx).cpu()
            orig_params = w1.numel()
            total_params += orig_params
            options = profile_layer_once(w1, name=name, index=idx, calib_data=calib_data,
                                         cr_candidates=cr_candidates, ks_ratio=ks_ratio)
            layer_profiles.append({
                'name': name,
                'idx': idx,
                'orig_params': orig_params,
                'options': options
            })
    return layer_profiles, total_params