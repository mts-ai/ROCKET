import json
from pathlib import Path

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def ensure_file_path(file_path: str) -> None:
    """
    Ensures that all parent directories of the given file path exist.
    If they don't, creates them recursively.
    
    Args:
        file_path (str): Full path to a file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def parse_and_save_results(res, output_path):

    filtered_results = {}
    
    for task_key, task_data in res.items():
        if "mmlu_" in task_key:
            continue
        if task_key == "wikitext":
            if "word_perplexity,none" in task_data:
                filtered_results[task_key] = {
                    "word_perplexity,none": round(task_data["word_perplexity,none"], 4)
                }
        elif task_key == "lambada_openai":
            task_result = {}
            if "acc,none" in task_data:
                task_result["acc,none"] = round(task_data["acc,none"], 4)
            if "acc_norm,none" in task_data:
                task_result["acc_norm,none"] = round(task_data["acc_norm,none"], 4)
            if "perplexity,none" in task_data:
                task_result["perplexity,none"] = round(task_data["perplexity,none"], 4)
            if task_result:
                filtered_results[task_key] = task_result
        else:
            task_result = {}
            if "acc,none" in task_data:
                task_result["acc,none"] = round(task_data["acc,none"], 4)
            if "acc_norm,none" in task_data:
                task_result["acc_norm,none"] = round(task_data["acc_norm,none"], 4)
            if task_result:
                filtered_results[task_key] = task_result

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(filtered_results, f, indent=2)