import argparse
import yaml
import os
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from rocket.data.prepare_data import prepare_data


def compute_ppl(max_length, stride, data, model, device):
    model.to(device)
    model = model.eval()
    seq_len = data.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            output = model(input_ids, labels=target_ids)

            neg_log_likelihood = output.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tokenizer = AutoTokenizer.from_pretrained(cfg['compression']['output_dir'])
    tokenizer.pad_token = "[PAD]"
    train_dataset, val_dataset, test_dataset, data_collator = prepare_data(cfg['evaluation_ppl']['dataset_name'], tokenizer,
                                                                           2048,
                                                                           "PATH TO C4")
    model = AutoModelForCausalLM.from_pretrained(cfg['compression']['output_dir'], device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True)
    print(compute_ppl(2048, 2048, test_dataset, model, "cuda"))
if __name__ == '__main__':
    main()