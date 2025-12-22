import numpy as np

def preprocess_layer_profiles(layer_profiles, reference_cr=0.2):
    errors_at_ref_cr = []
    for layer in layer_profiles:
        for cr_target, actual_cr, err, ks_ratio in layer['options']:
            if abs(cr_target - reference_cr) < 1e-6:
                errors_at_ref_cr.append(err)
                break
    if errors_at_ref_cr:
        avg_error = np.mean(errors_at_ref_cr) * 1.0
    else:
        all_errors = [err for layer in layer_profiles for (_, _, err) in layer['options'] if 0 <= err < 2.0]
        avg_error = np.percentile(all_errors, 95) if all_errors else 1.0

    print(f"[Preprocessing] Avg error at cr={reference_cr}: {avg_error:.4f}")

    filtered_profiles = []
    for layer in layer_profiles:
        filtered_options = [(cr, ac, err, ks) for cr, ac, err, ks in layer['options'] if (err <= avg_error and 0 <= err < 2.0) or (cr <= reference_cr)]
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