import math
import heapq
import math
try:
    from ortools.graph.python import min_cost_flow
except ImportError:
    # Fallback for older versions (unlikely, but safe)
    raise ImportError("Please install OR-Tools >= 9.0: pip install --upgrade ortools")
    
def solve_mckp_target_based(
    layer_profiles,
    error_lookup,
    kept_frac_lookup,
    ks_lookup,
    total_params,
    target_kept_ratio=0.8,
    param_precision=30000
):
    min_kept = target_kept_ratio * total_params
    scale = param_precision / total_params
    # Now each DP entry is: (error, cr_list, ks_list)
    dp = {0: (0.0, [], [])}

    for layer in layer_profiles:
        name = layer['name']
        idx = layer['idx']
        orig_params = layer['orig_params']
        candidates = []
        for cr_target in kept_frac_lookup[name][idx]:
            kept_frac = kept_frac_lookup[name][idx][cr_target]
            error = error_lookup[name][idx][cr_target]
            ks_ratio = ks_lookup[name][idx][cr_target]
            kept = kept_frac * orig_params
            if kept <= 0 or kept > orig_params or not (0 <= error < 2.0):
                continue
            candidates.append((kept, error, cr_target, ks_ratio))
        if not candidates:
            candidates = [(orig_params, 0.0, 1.0, 1.0)]

        new_dp = {}
        for kept_scaled, (err_so_far, cr_list, ks_list) in dp.items():
            for kept, err, cr_target, ks_ratio in candidates:
                new_kept = kept_scaled + kept * scale
                new_kept_int = int(round(new_kept))
                new_err = err_so_far + err
                new_cr_list = cr_list + [cr_target]
                new_ks_list = ks_list + [ks_ratio]
                if new_kept_int not in new_dp or new_err < new_dp[new_kept_int][0]:
                    new_dp[new_kept_int] = (new_err, new_cr_list, new_ks_list)
        dp = new_dp

        if len(dp) > 50000:
            items = sorted(dp.items())
            best_err = float('inf')
            pruned = {}
            for kept_int, (err, crs, kss) in reversed(items):
                if err < best_err:
                    pruned[kept_int] = (err, crs, kss)
                    best_err = err
            dp = pruned

    min_kept_scaled = min_kept * scale
    best_error = float('inf')
    best_allocation = None
    best_ks_allocation = None
    best_kept = 0.0
    for kept_scaled, (err, crs, kss) in dp.items():
        if kept_scaled >= min_kept_scaled - 1e-5:
            if err < best_error:
                best_error = err
                best_allocation = crs
                best_ks_allocation = kss
                best_kept = kept_scaled / scale

    if best_allocation is None:
        fallback_cr = target_kept_ratio
        best_allocation = [fallback_cr] * len(layer_profiles)
        best_ks_allocation = [1.0] * len(layer_profiles)  # fallback ks_ratio
        best_kept = target_kept_ratio * total_params
        best_error = sum(error_lookup[layer['name']][layer['idx']].get(fallback_cr, 0.1) for layer in layer_profiles)

    achieved_removed_ratio = 1.0 - (best_kept / total_params)
    return best_allocation, best_ks_allocation, best_error, achieved_removed_ratio

def solve_mckp_min_cost_flow(
    layer_profiles,
    error_lookup,
    kept_frac_lookup,
    ks_lookup,
    total_params,
    target_compression_ratio=0.2,
    param_precision=30000
):
    n_layers = len(layer_profiles)
    K_min = target_compression_ratio * total_params
    scale = param_precision / total_params
    K_min_scaled = int(math.ceil(K_min * scale - 1e-5))

    # Precompute candidates per layer: (scaled_kept, cost, cr_target, ks_ratio)
    layers = []
    for layer in layer_profiles:
        name = layer['name']
        idx = layer['idx']
        orig_params = layer['orig_params']
        candidates = []
        for cr_target in kept_frac_lookup[name][idx]:
            kept_frac = kept_frac_lookup[name][idx][cr_target]
            error = error_lookup[name][idx][cr_target]
            ks_ratio = ks_lookup[name][idx][cr_target]
            kept = kept_frac * orig_params
            if kept <= 0 or kept > orig_params or not (0 <= error < 2.0):
                continue
            scaled_kept = int(round(kept * scale))
            cost = int(round(error * 1_000_000))
            candidates.append((scaled_kept, cost, cr_target, ks_ratio))
        if not candidates:
            scaled_kept = int(round(orig_params * scale))
            cost = 0
            candidates = [(scaled_kept, cost, 1.0, 1.0)]  # include default ks_ratio
        layers.append(candidates)

    max_scaled = param_precision

    # Build state-to-node mapping
    node_index = {}
    next_id = 0

    def get_node_id(i, k):
        nonlocal next_id
        key = (i, k)
        if key not in node_index:
            node_index[key] = next_id
            next_id += 1
        return node_index[key]

    # Source and sink
    source = get_node_id(0, 0)
    sink = next_id
    next_id += 1

    # Initialize solver
    smcf = min_cost_flow.SimpleMinCostFlow()

    # Track current reachable states
    current_states = {0}

    # Build graph layer by layer
    for i in range(n_layers):
        next_states = set()
        for k_in in current_states:
            u = get_node_id(i, k_in)
            for scaled_kept, cost, cr, ks in layers[i]:
                k_out = k_in + scaled_kept
                if k_out > max_scaled:
                    continue
                v = get_node_id(i + 1, k_out)
                smcf.add_arc_with_capacity_and_unit_cost(u, v, 1, cost)
                next_states.add(k_out)
        current_states = next_states

    # Connect feasible terminal states to sink
    terminal_arcs = 0
    for k in current_states:
        if k >= K_min_scaled:
            u = get_node_id(n_layers, k)
            smcf.add_arc_with_capacity_and_unit_cost(u, sink, 1, 0)
            terminal_arcs += 1

    if terminal_arcs == 0:
        # Fallback
        fallback_cr = target_compression_ratio
        allocation = [fallback_cr] * n_layers
        allocations_ks = [1.0] * n_layers
        kept = target_compression_ratio * total_params
        error = sum(
            error_lookup[layer['name']][layer['idx']].get(fallback_cr, 0.1)
            for layer in layer_profiles
        )
        achieved_removed_ratio = 1.0 - (kept / total_params)
        return allocation, allocations_ks, error, achieved_removed_ratio

    # Set supply/demand
    smcf.set_node_supply(source, 1)
    smcf.set_node_supply(sink, -1)

    # Solve
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        # Fallback
        fallback_cr = target_compression_ratio
        allocation = [fallback_cr] * n_layers
        allocations_ks = [1.0] * n_layers  # <-- added
        kept = target_compression_ratio * total_params
        error = sum(
            error_lookup[layer['name']][layer['idx']].get(fallback_cr, 0.1)
            for layer in layer_profiles
        )
        achieved_removed_ratio = 1.0 - (kept / total_params)
        return allocation, allocations_ks, error, achieved_removed_ratio  # <-- updated return

    # Reconstruct path
    allocation = [None] * n_layers
    allocations_ks = [None] * n_layers  # <-- new list to track ks_ratios
    current_node = source
    current_k = 0

    # Build reverse lookup
    node_to_state = {v: (i, k) for (i, k), v in node_index.items()}

    for i in range(n_layers):
        found = False
        for arc in range(smcf.num_arcs()):
            if smcf.tail(arc) == current_node and smcf.flow(arc) == 1:
                head = smcf.head(arc)
                if head == sink:
                    continue
                if head not in node_to_state:
                    continue
                next_layer, next_k = node_to_state[head]
                if next_layer != i + 1:
                    continue
                scaled_kept = next_k - current_k
                # Match candidate by scaled_kept (since cost/CR may not be unique, but kept should be)
                for cand_kept, cand_cost, cr, ks in layers[i]:
                    if cand_kept == scaled_kept:
                        allocation[i] = cr
                        allocations_ks[i] = ks  # <-- record ks_ratio
                        found = True
                        break
                if found:
                    current_node = head
                    current_k = next_k
                    break
        if not found:
            # Fallback to first candidate
            allocation[i] = layers[i][0][2]
            allocations_ks[i] = layers[i][0][3]  # <-- fallback ks

    # Compute final metrics (ks_ratios not used here)
    total_kept = 0.0
    total_error = 0.0
    for i, layer in enumerate(layer_profiles):
        name, idx = layer['name'], layer['idx']
        cr = allocation[i]
        kept_frac = kept_frac_lookup[name][idx].get(cr)
        if kept_frac is None:
            kept_frac = cr
        kept = kept_frac * layer['orig_params']
        error = error_lookup[name][idx].get(cr, 0.1)
        total_kept += kept
        total_error += error

    achieved_removed_ratio = 1.0 - (total_kept / total_params)
    return allocation, allocations_ks, total_error, achieved_removed_ratio

import heapq

def solve_dijkstra(
    layer_profiles,
    error_lookup,
    kept_frac_lookup,
    ks_lookup,
    total_params,
    target_compression_ratio=0.8,
    param_precision=30000,
    error_scale_factor=1_000_000
):
    n_layers = len(layer_profiles)
    min_kept = target_compression_ratio * total_params
    scale = param_precision / total_params
    min_kept_scaled = min_kept * scale

    # Precompute candidates per layer: (scaled_kept, cost_int, cr_target, ks_ratio)
    layers = []
    for layer in layer_profiles:
        name = layer['name']
        idx = layer['idx']
        orig_params = layer['orig_params']
        candidates = []
        for cr_target in kept_frac_lookup[name][idx]:
            kept_frac = kept_frac_lookup[name][idx][cr_target]
            error = error_lookup[name][idx][cr_target]
            ks_ratio = ks_lookup[name][idx][cr_target]
            kept = kept_frac * orig_params
            if kept <= 0 or kept > orig_params or not (0 <= error < 2.0):
                continue
            scaled_kept = int(round(kept * scale))
            cost_int = int(round(error * error_scale_factor))
            candidates.append((scaled_kept, cost_int, cr_target, ks_ratio))
        if not candidates:
            scaled_kept = int(round(orig_params * scale))
            cost_int = 0
            candidates = [(scaled_kept, cost_int, 1.0, 1.0)]  # include default ks_ratio
        layers.append(candidates)

    max_scaled = param_precision

    # dist[(i, kept_scaled)] = (min_cost_int, cr_choices, ks_choices)
    dist = {}
    # Priority queue: (cost_int, i, kept_scaled, cr_choices, ks_choices)
    pq = []
    heapq.heappush(pq, (0, 0, 0, [], []))

    best_solution = None  # (cost_int, kept_scaled, cr_choices, ks_choices)

    while pq:
        cost_int, i, kept_scaled, cr_choices, ks_choices = heapq.heappop(pq)

        if (i, kept_scaled) in dist:
            continue
        dist[(i, kept_scaled)] = (cost_int, cr_choices, ks_choices)

        if i == n_layers:
            if kept_scaled >= min_kept_scaled - 1e-5:
                if best_solution is None or cost_int < best_solution[0]:
                    best_solution = (cost_int, kept_scaled, cr_choices, ks_choices)
            continue

        for scaled_kept, cand_cost, cr, ks in layers[i]:
            new_kept = kept_scaled + scaled_kept
            if new_kept > max_scaled:
                continue
            new_cost = cost_int + cand_cost
            new_cr_choices = cr_choices + [cr]
            new_ks_choices = ks_choices + [ks]
            if (i + 1, new_kept) not in dist:
                heapq.heappush(pq, (new_cost, i + 1, new_kept, new_cr_choices, new_ks_choices))

    # --- Handle solution or fallback ---
    if best_solution is not None:
        _, final_kept_scaled, allocation, allocations_ks = best_solution
        total_kept = 0.0
        total_error = 0.0
        for i, layer in enumerate(layer_profiles):
            name, idx = layer['name'], layer['idx']
            cr = allocation[i]
            kept_frac = kept_frac_lookup[name][idx].get(cr)
            if kept_frac is None:
                kept_frac = cr
            kept = kept_frac * layer['orig_params']
            error = error_lookup[name][idx].get(cr, 0.1)
            total_kept += kept
            total_error += error
    else:
        fallback_cr = target_compression_ratio
        allocation = [fallback_cr] * n_layers
        allocations_ks = [1.0] * n_layers  # fallback ks_ratio
        total_kept = fallback_cr * total_params
        total_error = sum(
            error_lookup[layer['name']][layer['idx']].get(fallback_cr, 0.1)
            for layer in layer_profiles
        )

    achieved_removed_ratio = 1.0 - (total_kept / total_params)
    return allocation, allocations_ks, total_error, achieved_removed_ratio

# allocation, total_err, achieved_removed = solve_dijkstra(
#     layer_profiles,
#     error_lookup,
#     kept_frac_lookup,
#     total_params,
#     target_compression_ratio=0.2,   # keep 80%
#     param_precision=30000
# )
