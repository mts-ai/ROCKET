import argparse
import yaml
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_USE_DISTRIBUTED"] = "false"
os.environ["ACCELERATE_CONFIG_FILE"] = ""

import lm_eval
from lm_eval.models.huggingface import HFLM
from swiftsvd.utils.io import parse_and_save_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["compression"]["output_dir"]
    res = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=cfg["evaluation"]["tasks"],
        num_fewshot=0,
        batch_size=cfg["evaluation"]["batch_size"],
        max_batch_size=cfg["evaluation"]["max_batch_size"],
        device=cfg["evaluation"]["device"]
    )

    # print(res["results"])
    parse_and_save_results(res["results"], cfg["evaluation"]['res_path'])

if __name__ == "__main__":
    main()