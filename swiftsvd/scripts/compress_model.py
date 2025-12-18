import argparse
import yaml
import os
import json
import shutil
from swiftsvd.utils.seed import seed_all
from swiftsvd.utils.io import load_json
from swiftsvd.profiling.postprocess import preprocess_layer_profiles, build_error_and_kept_lookup
from swiftsvd.compression.mckp import solve_mckp_target_based, solve_mckp_min_cost_flow, solve_dijkstra
from swiftsvd.compression.swiftsvd import svd_with_magnitude_sparsity_on_v
from swiftsvd.utils.model_utils import get_weight_transposed, compute_actual_compression
from swiftsvd.modeling.modeling_llama_svdllm import LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_all(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    cache = load_json(cfg["profiling"]["profile_cache"])
    layer_profiles_raw = cache["layer_profiles"]
    total_params = cache["total_params"]
    adam_refinement_step = cfg["compression"]["adam_refine_steps"]
    layer_profiles, _ = preprocess_layer_profiles(layer_profiles_raw, reference_cr=cfg["compression"]["target_kept_ratio"])
    error_lookup, kept_frac_lookup, ks_lookup = build_error_and_kept_lookup(layer_profiles)
    if cfg["compression"]["method"] == "knapsack":
        cr_allocation, ks_allocation, total_err, achieved_removed = solve_mckp_target_based(
            layer_profiles,
            error_lookup,
            kept_frac_lookup,
            ks_lookup,
            total_params,
            target_kept_ratio=cfg["compression"]["target_kept_ratio"],
            param_precision=cfg["compression"]["param_precision"]
        )
    
    elif cfg["compression"]["method"] == "maxflow":
        allocation, total_err, achieved_removed = solve_mckp_min_cost_flow(
            layer_profiles,
            error_lookup,
            kept_frac_lookup,
            total_params,
            target_compression_ratio=cfg["compression"]["target_kept_ratio"],
            param_precision=cfg["compression"]["param_precision"]
        )
    else:
        allocation, total_err, achieved_removed = solve_dijkstra(
            layer_profiles,
            error_lookup,
            kept_frac_lookup,
            total_params,
            target_compression_ratio=cfg["compression"]["target_kept_ratio"],
            param_precision=cfg["compression"]["param_precision"]
        )
    

    cr_nested = {}
    for layer, cr_chosen, ks_chosen in zip(layer_profiles, cr_allocation, ks_allocation):
        name = layer['name']
        idx = layer['idx']
        if name not in cr_nested:
            cr_nested[name] = {}
        cr_nested[name][idx] = {'cr': cr_chosen, 'ks': ks_chosen}

    print(f"Achieved compression (removed): {achieved_removed:.3f}")
    with open(cfg["profiling"]["cr_cache"],"w") as f:
        json.dump(cr_nested, f)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], device_map="cpu", torch_dtype=getattr(torch, cfg["model"]["dtype"])
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    new_model = LlamaForCausalLM.from_pretrained(
        cfg["model"]["name"], device_map="cpu", torch_dtype=getattr(torch, cfg["model"]["dtype"]), compression_path=cfg["profiling"]["cr_cache"]
    )
    module_names = cfg["profiling"]["module_names"]
    for name in module_names:
        for i in range(len(model.model.layers)):
            w_orig = get_weight_transposed(model, name, i)
            u, v = svd_with_magnitude_sparsity_on_v(
                w_orig,
                name=name,
                index=i,
                cr_ks=cr_nested[name][i],
                calib_data=cfg["calib"]["data_path"],
                dobi_like=cfg["compression"]["dobi_like"],
                adam_refine_steps = adam_refinement_step
            )
            with torch.no_grad():
                w_recon = (u @ v).cpu()
                model_weight_name = name.split('.')[-1]
                if "mlp" in name:
                    getattr(model.model.layers[i].mlp, model_weight_name).weight.copy_(w_recon.T)
                    getattr(new_model.model.layers[i].mlp, model_weight_name + "_u").weight.copy_(u.T)
                    getattr(new_model.model.layers[i].mlp, model_weight_name + "_v").weight.copy_(v.T)
                else:
                    getattr(model.model.layers[i].self_attn, model_weight_name).weight.copy_(w_recon.T)
                    getattr(new_model.model.layers[i].self_attn, model_weight_name + "_u").weight.copy_(u.T)
                    getattr(new_model.model.layers[i].self_attn, model_weight_name + "_v").weight.copy_(v.T)
            del u, v
            gc.collect()

    out_dir = cfg["compression"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    new_model.save_pretrained(out_dir + "_sep")
    tokenizer.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir + "_sep")

    sep_dir = out_dir + "_sep"
    
    # 1. Copy and rename the modeling file THIS SHOULD BE MODIFIED LATER TO SUPPORT OTHER MODELS
    src_modeling_file = "swiftsvd/modeling/modeling_llama_svdllm.py"  # adjust if path differs
    dst_modeling_file = os.path.join(sep_dir, "modeling_llama_svd.py")
    shutil.copyfile(src_modeling_file, dst_modeling_file)
    
    # 2. Load the config.json from the sep checkpoint
    config_path = os.path.join(sep_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 3. Inject auto_map
    config["auto_map"] = {
        "AutoModel": "modeling_llama_svd.LlamaModel",
        "AutoModelForCausalLM": "modeling_llama_svd.LlamaForCausalLM"
    }
    
    # 4. Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    # Final compression report
    total_orig, total_kept, actual_cr = compute_actual_compression(cr_nested, d=2048, num_layers=16)
    print(f"Final compression: {actual_cr:.3f} (removed)")

if __name__ == "__main__":
    main()
