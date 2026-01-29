import argparse
import yaml
import os
import torch
from rocket.utils.seed import seed_all
from rocket.data.prepare_data import prepare_data
from rocket.calib.calib import Calib
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset, DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_all(42)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    calib_path = cfg["calib"]["data_path"]
    if os.path.exists(calib_path):
        print(f"✅ Calibration data already exists at {calib_path}")
        return

    os.makedirs(os.path.dirname(calib_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], device_map="cuda", torch_dtype=getattr(torch, cfg["model"]["dtype"])
    )

    train_dataset, _, _, data_collator = prepare_data(
        cfg["calib"]["dataset"], tokenizer, cfg["calib"]["seq_len"], None
    )

    torch.manual_seed(cfg["calib"]["seed"])
    indices = torch.randperm(len(train_dataset))[:cfg["calib"]["num_samples"]]
    subset = Subset(train_dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=cfg["calib"]["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )

    calib_names = ["self_attn.k_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]
    Calib.build_calibration_dataset(model, dataloader, calib_names, "llama3", calib_path)
    print("✅ Calibration data built.")

if __name__ == "__main__":
    main()
