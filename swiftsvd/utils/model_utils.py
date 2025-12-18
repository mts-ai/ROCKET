import torch
import json
import os
import gc

def frobenius_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same dimensions")
    norm_B = torch.linalg.norm(B, 'fro') + 1e-12
    return torch.linalg.norm(A - B, 'fro') / norm_B

def get_weight_transposed(model, name, layer_idx):
    if "mlp" in name:
        proj = name.split('.')[-1]
        return getattr(model.model.layers[layer_idx].mlp, proj).weight.T
    else:
        proj = name.split('.')[-1]
        return getattr(model.model.layers[layer_idx].self_attn, proj).weight.T

def get_cr(d1, d2, L, k, sparsity):
    vanilla = d1 * d2 * L
    d = d1 * k
    c = sparsity * d2 * L
    return 1 - (d + c) / vanilla

def get_k_and_sparsity(cr, d1, d2, L, ks_ratio):
    vanilla = d1 * d2 * L
    k = (1 - cr) * d1 * d2 * L / (d1 + d2 * L / ks_ratio)
    sparsity = k / ks_ratio
    return int(k), 1 - int(sparsity) / (int(k) + 1e-8)

def compute_actual_compression(cr_nested, d, num_layers):
    attn_params = d * d
    mlp_gate_up = d * (4 * d)
    mlp_down = (4 * d) * d

    total_orig = 0
    total_kept = 0

    for layer_idx in range(num_layers):
        for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']:
            cr = cr_nested[proj][layer_idx]['cr']
            orig_p = attn_params
            kept_p = cr * orig_p
            total_orig += orig_p
            total_kept += kept_p
        for proj in ['mlp.gate_proj', 'mlp.up_proj']:
            cr = cr_nested[proj][layer_idx]['cr']
            orig_p = mlp_gate_up
            kept_p = cr * orig_p
            total_orig += orig_p
            total_kept += kept_p
        proj = 'mlp.down_proj'
        cr = cr_nested[proj][layer_idx]['cr']
        orig_p = mlp_down
        kept_p = cr * orig_p
        total_orig += orig_p
        total_kept += kept_p

    actual_compression_ratio = 1 - (total_kept / total_orig)
    return total_orig, total_kept, actual_compression_ratio