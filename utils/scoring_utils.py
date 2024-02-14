import sys
from utils.config import args
if "llama" in args.generator:
    from scorer.llama_scorer import Llama_Scorer
    Scorer = Llama_Scorer
elif "mistral" in args.generator:
    from scorer.mistral_scorer import Mistral_Scorer
    Scorer = Mistral_Scorer
import tqdm
import torch
# Thanks to DPO for their easy-to-use code. We have modified it to suit our needs.
# Reference: https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py

def tokenize_for_fair_scoring(
    context: str, 
    response: str, 
    tokenizer
):
    response_tokens = tokenizer(response, add_special_tokens=False)
    context_tokens = tokenizer(context)

    assert tokenizer.eos_token_id not in response_tokens['input_ids'], f"Response contains EOS token: {response}"

    response_tokens['input_ids'].append(tokenizer.eos_token_id)
    response_tokens['attention_mask'].append(1)
    response_tokens.pop('token_type_ids', None)
    if "token_type_ids" in response_tokens:
        response_tokens['token_type_ids'].append(0)
    
    # Create labels
    sequence_tokens = {k: context_tokens[k] + response_tokens[k] for k in response_tokens}
    sequence_tokens['labels'] = sequence_tokens['input_ids'][:]
    sequence_tokens['labels'][:len(context_tokens['input_ids'])] = [-100] * len(context_tokens['input_ids'])

    length = len(sequence_tokens['input_ids'])
    assert length == len(sequence_tokens['labels']), f"Input and label length mismatch: {sequence_tokens['input_ids']} vs {sequence_tokens['labels']}"
    assert length == len(sequence_tokens['attention_mask']), f"Input and attention mask length mismatch: {sequence_tokens['input_ids']} vs {sequence_tokens['attention_mask']}"
    
    return sequence_tokens

def padding(batch, tokenizer):
    input_batch = {}
    for sample in batch:
        for k in sample.keys():
            if k not in input_batch:
                input_batch[k] = []
            input_batch[k].append(sample[k])

    pad_ids = {
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
        "token_type_ids": 0,
        "labels": -100,
    }

    # padding
    for k in input_batch.keys():
        max_length = max([len(x) for x in input_batch[k]])
        for i in range(len(input_batch[k])):
            input_batch[k][i] += [pad_ids[k]] * (max_length - len(input_batch[k][i]))
        input_batch[k] = torch.tensor(input_batch[k])
    
    return input_batch

def collate_fn(batch_inputs, tokenizer):
    # inputs = [
    #     {
    #         "context": "",
    #         "responses": [],
    #     },
    # ]
    num_generation = len(batch_inputs[0]["responses"])
    batch = []
    for inputs in batch_inputs:
        for i in range(num_generation):
            sample = {
                "context": inputs["context"],
                "response": inputs["responses"][i],
            }
            sample = tokenize_for_fair_scoring(
                sample["context"],
                sample["response"],
                tokenizer,
            )
            batch.append(sample)
    padded_batch = padding(batch, tokenizer)
    for k in padded_batch.keys():
        padded_batch[k] = padded_batch[k].view(-1, num_generation, padded_batch[k].shape[-1])
    
    return padded_batch

def score(
    data_manager,
    pos_mode,
    neg_mode,
    pos_model_direction,
    neg_model_direction,
):
    scorer = Scorer(args.model_name_or_path)
    tokenizer = scorer.tokenizer
    
    mode_roles = {"pos": (pos_mode, pos_model_direction), "neg": (neg_mode, neg_model_direction)}
    pred_res = {}
    if mode_roles["pos"][0] is None and mode_roles["neg"][0] is None:
        return None
    if not (mode_roles["pos"][0] is None):
        pred_res["pos"] = []
    if not (mode_roles["neg"][0] is None):
        pred_res["neg"] = []

    for mode in ["pos", "neg"]:
        if mode_roles[mode][0] == None:
            continue

        batch = []
        if args.batch_size > args.num_generation:
            batch_size = args.batch_size // args.num_generation
        else:
            batch_size = args.batch_size
        pbar = tqdm.tqdm(total=len(data_manager.test_set), desc=f"Scoring {mode}...")
        for sample_index, sample in enumerate(data_manager.test_set):
            batch.append(sample)
            if len(batch) == batch_size or sample_index == len(data_manager.test_set) - 1:
                batch = [
                    {
                        "context": data_manager.get_context(
                            sample,
                            mode = mode_roles[mode][0],
                            pos_or_neg = mode_roles[mode][1],
                            num_demonstrations = args.num_demonstrations,
                        ),
                        "responses": [
                            " " + (sample["candidates"][i]["t"].lstrip())
                            for i in range(args.num_generation)
                        ],
                    }
                    for sample in batch
                ]
                padded_batch = collate_fn(
                    batch,
                    tokenizer,
                )
                pred_scores = scorer.score(
                    padded_batch,
                    scorer_config = {},
                ) # [[score1, score2, score3], [score1, score2, score3], ...]
                pred_res[mode].extend(pred_scores)
                pbar.update(len(batch))
                batch = []
        pbar.close()
    return pred_res