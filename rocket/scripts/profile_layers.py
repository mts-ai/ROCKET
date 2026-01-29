import argparse
import yaml
import os
from rocket.utils.seed import seed_all
from rocket.profiling.profiler import profile_all_layers
from rocket.utils.io import save_json, load_json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_path = cfg["profiling"]["profile_cache"]
    if os.path.exists(cache_path):
        print(f"✅ Layer profiles found at {cache_path}")
        return

    seed_all(42)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], device_map="cuda", torch_dtype=getattr(torch, cfg["model"]["dtype"])
    )

    layer_profiles, total_params = profile_all_layers(
        model,
        cfg["profiling"]["module_names"],
        cfg["calib"]["data_path"],
        cfg["profiling"]["cr_candidates"],
        ks_ratios=cfg["profiling"]["ks_ratios"]
    )

    total_params = sum(p.numel() for layer in model.model.layers for p in layer.parameters())
    save_json({"layer_profiles": layer_profiles, "total_params": total_params}, cache_path)
    print(f"✅ Profiles saved to {cache_path}")

if __name__ == "__main__":
    main()