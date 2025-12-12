def solve_mckp_target_based(
    layer_profiles,
    error_lookup,
    kept_frac_lookup,
    total_params,
    target_kept_ratio=0.8,
    param_precision=30000
):
    min_kept = target_kept_ratio * total_params
    scale = param_precision / total_params
    dp = {0: (0.0, [])}

    for layer in layer_profiles:
        name = layer['name']
        idx = layer['idx']
        orig_params = layer['orig_params']
        candidates = []
        for cr_target in kept_frac_lookup[name][idx]:
            kept_frac = kept_frac_lookup[name][idx][cr_target]
            error = error_lookup[name][idx][cr_target]
            kept = kept_frac * orig_params
            if kept <= 0 or kept > orig_params or not (0 <= error < 2.0):
                continue
            candidates.append((kept, error, cr_target))
        if not candidates:
            candidates = [(orig_params, 0.0, 1.0)]

        new_dp = {}
        for kept_scaled, (err_so_far, cr_list) in dp.items():
            for kept, err, cr_target in candidates:
                new_kept = kept_scaled + kept * scale
                new_kept_int = int(round(new_kept))
                new_err = err_so_far + err
                if new_kept_int not in new_dp or new_err < new_dp[new_kept_int][0]:
                    new_dp[new_kept_int] = (new_err, cr_list + [cr_target])
        dp = new_dp

        if len(dp) > 50000:
            items = sorted(dp.items())
            best_err = float('inf')
            pruned = {}
            for kept_int, (err, crs) in reversed(items):
                if err < best_err:
                    pruned[kept_int] = (err, crs)
                    best_err = err
            dp = pruned

    min_kept_scaled = min_kept * scale
    best_error = float('inf')
    best_allocation = None
    best_kept = 0.0
    for kept_scaled, (err, crs) in dp.items():
        if kept_scaled >= min_kept_scaled - 1e-5:
            if err < best_error:
                best_error = err
                best_allocation = crs
                best_kept = kept_scaled / scale

    if best_allocation is None:
        fallback_cr = target_kept_ratio
        best_allocation = [fallback_cr] * len(layer_profiles)
        best_kept = target_kept_ratio * total_params
        best_error = sum(error_lookup[layer['name']][layer['idx']].get(fallback_cr, 0.1) for layer in layer_profiles)

    achieved_removed_ratio = 1.0 - (best_kept / total_params)
    return best_allocation, best_error, achieved_removed_ratio