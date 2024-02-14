import random
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append("..")
from retriever.bm25 import bm25_retrieval
from transformers import AutoTokenizer, AutoModel
import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
setup_seed(42)
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('../../models/all-mpnet-base-v2')
model = AutoModel.from_pretrained('../../models/all-mpnet-base-v2').to('cuda:1')

def get_sentence_embeddings(sentences):
    # sentences: list of strings
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(model.device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1) # [n_samples, embed_size]

    return sentence_embeddings

def get_similarity(queries, docss):
    # query: string
    # docs: list of strings
    batch = []
    similarity_matrix = []
    batch_size = 2
    n_docs = len(docss[0])
    
    pbar = tqdm.tqdm(total=len(queries))
    for index, qds_tuple in enumerate(zip(queries, docss)):
        query, docs = qds_tuple
        sentences = [query] + docs
        batch.append(sentences) # [batch_size, n_docs+1]
        if index % batch_size == 0 or index == len(queries) - 1:
            local_batch_size = len(batch)
            batch = sum(batch, []) # [batch_size * (n_docs+1)]
            embeddings = get_sentence_embeddings(batch).view(local_batch_size, n_docs+1, -1) # [batch_size, (n_docs+1), embed_size]
            local_similarity_matrix = torch.cosine_similarity(embeddings[:, 0, :].unsqueeze(1), embeddings[:, 1:], dim=2) # [batch_size, embed_size] * [batch_size, n_docs, embed_size] -> [batch_size, n_docs]
            similarity_matrix.append(local_similarity_matrix.cpu().tolist())
            pbar.update(local_batch_size)
            batch = []
        
    similarity_matrix = sum(similarity_matrix, []) # [n_queries, n_docs]
    pbar.close()
    return similarity_matrix

def sbert_retrieval(
    trainset,
    num_demonstrations,
    test_samples,
):
    demos = bm25_retrieval(
        trainset,
        num_demonstrations,
        test_samples,
    )
    test_samples = [sample["prompt"].strip() for sample in test_samples]
    demo_samples = [[sample["prompt"].strip() for sample in demo] for demo in demos]
    similarity_matrix = get_similarity(test_samples, demo_samples) # [n_queries, n_docs]
    sorted_indices = np.argsort(-np.array(similarity_matrix), axis=1) # [n_queries, n_docs] # descending order
    new_demos = []
    for i in range(len(test_samples)):
        new_demos.append([demos[i][index] for index in sorted_indices[i][:num_demonstrations]])
    return new_demos

def sbert_retrieval_with_bm25_res(
    demos,
    num_demonstrations,
    test_samples,
):
    test_samples = [sample["prompt"].strip() for sample in test_samples]
    demo_samples = [[sample["prompt"].strip() for sample in demo] for demo in demos]
    similarity_matrix = get_similarity(test_samples, demo_samples) # [n_queries, n_docs]
    sorted_indices = np.argsort(-np.array(similarity_matrix), axis=1) # [n_queries, n_docs] # descending order
    new_demos = []
    for i in range(len(test_samples)):
        new_demos.append([demos[i][index] for index in sorted_indices[i][:num_demonstrations]])
    return new_demos, sorted_indices