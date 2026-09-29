import argparse
import yaml
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_USE_DISTRIBUTED"] = "false"
os.environ["ACCELERATE_CONFIG_FILE"] = ""

import lm_eval
from lm_eval.models.huggingface import HFLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["compression"]["output_dir"]
    model_args = f"pretrained={model_path}"
    # Optional context cap: rolling-window tasks (wikitext) otherwise use the model's full context, which
    # does not fit in memory on MPS for 32k-context models.
    if cfg["evaluation"].get("max_length"):
        model_args += f",max_length={cfg['evaluation']['max_length']}"
    res = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=cfg["evaluation"]["tasks"],
        num_fewshot=0,
        batch_size=cfg["evaluation"]["batch_size"],
        max_batch_size=cfg["evaluation"]["max_batch_size"],
        device=cfg["evaluation"]["device"]
    )
    print(res["results"])

if __name__ == "__main__":
    main()