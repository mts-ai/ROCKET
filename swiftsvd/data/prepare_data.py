from functools import partial
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


def tokenize_func(example, tokenizer, content):
    return tokenizer(example[content])


def group_text(examples, context_length):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // context_length) * context_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + context_length] for i in range(0, total_length, context_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def prepare_data(dataset_name, tokenizer, context_length, dataset_cache_dir=None):
    if dataset_name == "wikitext":
        train_dataset, val_dataset, tokenized_test, data_collator = prep_wikitext_2_raw_v1(context_length, tokenizer,
                                                                                               dataset_cache_dir)
    elif dataset_name =="refinedweb":
        train_dataset, val_dataset, tokenized_test, data_collator = prep_refined_web(context_length, tokenizer,
                                                                                               dataset_cache_dir)
    elif dataset_name == "ptb":
        train_dataset, val_dataset, tokenized_test, data_collator = prep_ptb(context_length, tokenizer,
                                                                             dataset_cache_dir)
    elif dataset_name == "c4":
        train_dataset, val_dataset, tokenized_test, data_collator = prep_c4(context_length, tokenizer,
                                                                            dataset_cache_dir="./")
    elif dataset_name == "alpaca":
        train_dataset, val_dataset, tokenized_test, data_collator = prep_alpaca(context_length, tokenizer,
                                                                                dataset_cache_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return train_dataset, val_dataset, tokenized_test, data_collator


def prep_wikitext_2_raw_v1(context_length, tokenizer, dataset_cache_dir=None):
    print("load wikitext dataset")
    train_raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                                     dataset_cache_dir=dataset_cache_dir)
    val_raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                                   dataset_cache_dir=dataset_cache_dir)
    test_raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", dataset_cache_dir=dataset_cache_dir)
    print(train_raw_dataset)
    func = partial(tokenize_func, tokenizer=tokenizer, content="text")
    tokenized_train = train_raw_dataset.map(func, num_proc=4, batched=True, remove_columns="text")
    tokenized_val = val_raw_dataset.map(func, num_proc=4, batched=True, remove_columns="text")
    tokenized_test = tokenizer("\n\n".join(test_raw_dataset["text"]), return_tensors="pt")

    func = partial(group_text, context_length=context_length)
    train_dataset = tokenized_train.map(func, num_proc=4, batch_size=1024, batched=True)
    val_dataset = tokenized_val.map(func, num_proc=4, batch_size=1024, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    return train_dataset, val_dataset, tokenized_test, data_collator

def prep_refined_web(context_length, tokenizer, dataset_cache_dir=None):
    print("load refined_web dataset")
    
    # Load refined_web splits (adjust dataset name if needed, e.g., "togethercomputer/RedPajama-Data-1T")
    train_raw_dataset = load_dataset("HuggingFaceFW/fineweb","sample-10BT", split="train", dataset_cache_dir=dataset_cache_dir).select(range(500))
    val_raw_dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",split="train", dataset_cache_dir=dataset_cache_dir).select(range(500))
    test_raw_dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",split="train", dataset_cache_dir=dataset_cache_dir).select(range(500))
    print(train_raw_dataset)
    func = partial(tokenize_func, tokenizer=tokenizer, content="text")
    tokenized_train = train_raw_dataset.map(func, num_proc=4, batched=True, remove_columns=['text','id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'])
    tokenized_val = val_raw_dataset.map(func, num_proc=4, batched=True, remove_columns=['text','id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'])
    tokenized_test = tokenizer("\n\n".join(test_raw_dataset["text"]), return_tensors="pt")

    func = partial(group_text, context_length=context_length)
    train_dataset = tokenized_train.map(func, num_proc=4, batch_size=1024, batched=True)
    val_dataset = tokenized_val.map(func, num_proc=4, batch_size=1024, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    return train_dataset, val_dataset, tokenized_test, data_collator

def prep_ptb(context_length, tokenizer, dataset_cache_dir=None):
    print("load ptb dataset")
    train_raw_dataset = load_dataset("ptb_text_only", "penn_treebank", split='train',
                                     dataset_cache_dir=dataset_cache_dir)
    val_raw_dataset = load_dataset("ptb_text_only", "penn_treebank", split='validation',
                                   dataset_cache_dir=dataset_cache_dir)
    test_raw_dataset = load_dataset("ptb_text_only", "penn_treebank", split='test',
                                    dataset_cache_dir=dataset_cache_dir)
    func = partial(tokenize_func, tokenizer=tokenizer, content="sentence")
    tokenized_train_data = train_raw_dataset.map(func, num_proc=4, batched=True, remove_columns="sentence")
    tokenized_val_data = val_raw_dataset.map(func, num_proc=4, batched=True, remove_columns="sentence")
    tokenized_test_data = tokenizer("\n\n".join(test_raw_dataset['sentence']), return_tensors="pt")

    func = partial(group_text, context_length=context_length)
    train_dataset = tokenized_train_data.map(func, num_proc=4, batch_size=1024, batched=True)
    val_dataset = tokenized_val_data.map(func, num_proc=4, batch_size=1024, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    return train_dataset, val_dataset, tokenized_test_data, data_collator


def prep_c4(context_length, tokenizer, dataset_cache_dir=None):
    print("load C4 dataset")
    train_raw_dataset = load_dataset("json", data_dir=dataset_cache_dir, data_files="c4-validation.json")['train']
    val_raw_dataset = load_dataset("json", data_dir=dataset_cache_dir, data_files="c4-validation.json")['train']
    test_raw_dataset = val_raw_dataset

    tokenized_test_data = tokenizer("\n\n".join(test_raw_dataset['text'][0:2000]), return_tensors="pt")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    train_dataset = None
    val_dataset = None
    return train_dataset, val_dataset, tokenized_test_data, data_collator


def prep_alpaca(context_length, tokenizer, dataset_cache_dir=None):
    print("load Alpaca dataset")
    train_raw_dataset = load_dataset("tatsu-lab/alpaca", split='train',
                                     dataset_cache_dir=dataset_cache_dir)

    func = partial(tokenize_func, tokenizer=tokenizer, content="text")
    tokenized_train_data = train_raw_dataset.map(func, num_proc=1, batched=True,
                                                 remove_columns=["text", "instruction", "input", "output"])

    func = partial(group_text, context_length=context_length)
    train_dataset = tokenized_train_data.map(func, num_proc=1, batch_size=1024, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    return train_dataset, None, None, data_collator
