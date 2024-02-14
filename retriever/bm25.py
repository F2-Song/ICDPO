from gensim.summarization.bm25 import BM25
import random
random.seed(42)
from transformers import AutoTokenizer
import numpy as np
import tqdm

tokenizer = AutoTokenizer.from_pretrained("llama-7b")
def bm25_retrieval(
    trainset,
    num_demonstrations,
    test_samples,
    window_size=30,
):
    tokenized_trainset = []
    for sample in trainset:
        tokenized_sample = tokenizer.convert_ids_to_tokens(tokenizer.encode(sample["prompt"].strip(), add_special_tokens=False))
        tokenized_sample = tokenized_sample[-window_size:]
        tokenized_trainset.append(tokenized_sample)
    
    tokenized_testset = []
    for sample in test_samples:
        tokenized_sample = tokenizer.convert_ids_to_tokens(tokenizer.encode(sample["prompt"].strip(), add_special_tokens=False))
        tokenized_sample = tokenized_sample[-window_size:]
        tokenized_testset.append(tokenized_sample)

    bm25 = BM25(tokenized_trainset)
    score_matrix = []
    for sample in tqdm.tqdm(tokenized_testset):
        scores = bm25.get_scores(sample)
        scores = list(scores)
        score_matrix.append(scores)
    score_matrix = np.array(score_matrix)
    sorted_indices = np.argsort(-score_matrix, axis=1)
    
    demos = []
    for i in range(len(test_samples)):
        demos.append([trainset[int(index)] for index in sorted_indices[i][:num_demonstrations]])
    return demos