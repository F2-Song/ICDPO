import json
import os
import torch
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def load_raw_dataset(path):
    with open(path, 'r', encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f.readlines()]
    return dataset

def save_dataset(dataset, path, flag="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, flag, encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False)+"\n")

def early_truncation(text, stop_sequences):
    for stop in stop_sequences:
        stop_ix = text.find(stop)
        if stop_ix >= 0:
            text = text[:stop_ix]
    return text