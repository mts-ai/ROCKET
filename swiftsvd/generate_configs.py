# generate_configs.py
import yaml
import argparse
from pathlib import Path

CALIB_PATH = {
    "unsloth/Llama-3.2-1B": "/share/a.ammar/sparsesvd/llama1b/data/calib/refinedweb",
    "unsloth/llama-3-8b": "/share/b.mohammad/sparsesvd/llama8b/data/calib/refinedweb",
}

def sanitize_model_name(name: str) -> str:
    return name.replace("/", "__")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--target_kept_ratio", type=float, required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--results_base_dir", default="./results")
    parser.add_argument("--results_model_dir", default="/share/b.mohammad/swift_svd_final")
    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        cfg = yaml.safe_load(f)

    # Update model
    cfg["model"]["name"] = args.model_name
    cfg["calib"]["data_path"] = CALIB_PATH[args.model_name]
    # Update compression
    cfg["compression"]["target_kept_ratio"] = args.target_kept_ratio

    # Derive paths
    safe_model = sanitize_model_name(args.model_name)
    ratio_str = f"{args.target_kept_ratio:.2f}".replace(".", "_")
    exp_name = f"{safe_model}_cr{ratio_str}"
    output_dir = Path(args.results_base_dir) / exp_name
    model_dir = Path(args.results_model_dir) / exp_name
    cfg["compression"]["output_dir"] = str(model_dir)

    # Update cache paths (relative to output_dir or results_base_dir)
    cfg["profiling"]["profile_cache"] = str(output_dir / "layer_prof.json")
    cfg["profiling"]["cr_cache"] = str(output_dir / "cr_layer_prof.json")
    cfg["evaluation"]["res_path"] = str(output_dir / "results.json")
    # Ensure parent of output_config exists
    Path(args.output_config).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main()