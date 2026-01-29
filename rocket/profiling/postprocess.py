import numpy as np

def preprocess_layer_profiles(layer_profiles, reference_cr=0.2, alpha=1.0):
    errors_at_ref_cr = []
    for layer in layer_profiles:
        for cr_target, actual_cr, err, ks_ratio in layer['options']:
            if abs(cr_target - reference_cr) < 1e-6:
                errors_at_ref_cr.append(err)
                break
    if errors_at_ref_cr:
        avg_error = np.mean(errors_at_ref_cr) * alpha
    else:
        all_errors = [err for layer in layer_profiles for (_, _, err) in layer['options'] if 0 <= err < 2.0]
        avg_error = np.percentile(all_errors, 95) if all_errors else 1.0

    print(f"[Preprocessing] Avg error at cr={reference_cr}: {avg_error:.4f}")

    filtered_profiles = []
    for layer in layer_profiles:
        filtered_options = [(cr, ac, err, ks) for cr, ac, err, ks in layer['options'] if err <= avg_error and 0 <= err < 2.0]
        if not filtered_options:
            best_option = min(layer['options'], key=lambda x: x[2])
            filtered_options = [best_option]
            print(f"[Warning] Layer {layer['name']}[{layer['idx']}] had no valid options; kept best (err={best_option[2]:.4f})")
        filtered_profiles.append({
            'name': layer['name'],
            'idx': layer['idx'],
            'orig_params': layer['orig_params'],
            'options': filtered_options
        })
    return filtered_profiles, avg_error

def build_error_and_kept_lookup(layer_profiles):
    error_lookup = {}
    kept_frac_lookup = {}
    ks_lookup = {}
    for layer in layer_profiles:
        name = layer['name']
        idx = layer['idx']
        error_lookup.setdefault(name, {}).setdefault(idx, {})
        kept_frac_lookup.setdefault(name, {}).setdefault(idx, {})
        ks_lookup.setdefault(name, {}).setdefault(idx, {})
        for cr_target, actual_cr, err, ks_ratio in layer['options']:
            error_lookup[name][idx][cr_target] = err
            kept_frac_lookup[name][idx][cr_target] = cr_target
            ks_lookup[name][idx][cr_target] = ks_ratio
    return error_lookup, kept_frac_lookup, ks_lookup

import json
from typing import Dict, List, Optional

def find_min_alpha_for_target_cr(
    profile_dict: Dict,
    target_cr: float,
    alphas: List[float] = None
) -> Optional[float]:
    """
    Finds the smallest alpha ∈ [1.0, 1.1, ..., 2.0] such that:
      - For each layer, keep only options with error <= avg_error * alpha.
      - Choose, for each layer, the option with the *highest compression ratio (cr)*
        among the allowed ones (i.e., remove as much as possible while staying under error threshold).
      - The resulting *global compression ratio* (fraction of total params removed)
        is at least `target_cr`.
    """
    if alphas is None:
        alphas = [round(1.0 + i * 0.1, 1) for i in range(11)]  # 1.0 to 2.0

    layer_profiles = profile_dict["layer_profiles"]
    total_params = float(profile_dict["total_params"])


    errors_at_target = []
    for layer in layer_profiles:
        options = layer["options"]  # [cr, _, error, _]
        # Find option with cr closest to target_cr
        opt = min(options, key=lambda x: abs(x[0] - target_cr))
        errors_at_target.append(opt[2])
    avg_error = sum(errors_at_target) / len(errors_at_target)
    # Step 2: Try increasing alphas
    for alpha in alphas:
        error_threshold = avg_error * alpha
        total_retained = 0.0
        feasible = True

        for layer in layer_profiles:
            orig_params = float(layer["orig_params"])
            options = layer["options"]

            # Keep only options where error <= threshold
            valid_opts = [opt for opt in options if opt[2] <= error_threshold]
            if not valid_opts:
                best_by_error = min(options, key=lambda x: x[2])
                valid_opts = [best_by_error]

            # Among valid options, pick the one with *highest cr* (remove most params)
            best_opt = max(valid_opts, key=lambda x: x[0])  # max compression (max cr)
            cr = best_opt[0]  # fraction removed
            retained_frac = 1.0 - cr
            total_retained += orig_params * retained_frac

        if not feasible:
            continue

        global_cr = 1.0 - (total_retained / total_params)
        if global_cr >= target_cr - 1e-6:  # tolerance for floating point
            return alpha

    return None