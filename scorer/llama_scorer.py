import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import torch
import torch.nn.functional as F

class Llama_Scorer:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": self.tokenizer.eos_token,
            }
        )
        # use bf16
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16).eval()

    def batch_scoring(self, padded_batch, scorer_config):
        input_batch = {
            key: padded_batch[key] for key in padded_batch if key != "labels"
        } # [batch * num_generation, seq_len]

        with torch.no_grad():
            local_outputs = self.model(**input_batch, output_hidden_states=True, return_dict=True)
            local_logits = local_outputs.logits.to(torch.float32) #[batch * num_generation, seq_len, token_num]
            local_labels = padded_batch["labels"] #[batch * num_generation, seq_len]
            
            shift_logits = local_logits[..., :-1, :].contiguous() #[batch * num_generation, seq_len-1, token_num]
            shift_logits = F.log_softmax(shift_logits, dim=2) #[batch * num_generation, seq_len-1, token_num]
            
            shift_labels = local_labels[..., 1:] #[batch * num_generation, seq_len-1]
            label_mask = (shift_labels != -100) #[batch * num_generation, seq_len-1]
            shift_labels[shift_labels == -100] = 0
            per_token_logps = torch.gather(shift_logits, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2) #[batch * num_generation, seq_len-1]
            
            sum_scores = (per_token_logps * label_mask).sum(-1) #[batch * num_generation]

        return sum_scores.cpu().detach().numpy().tolist()
    
    def score(
        self, 
        padded_batch, 
        scorer_config
    ):
        # padded_batch = [batch, num_generation, seq_len]
        seq_len = padded_batch["input_ids"].shape[-1]
        num_generation = padded_batch["input_ids"].shape[1]
        for key in padded_batch:
            padded_batch[key] = padded_batch[key].to(self.model.device)
            padded_batch[key] = padded_batch[key].view(-1, seq_len) # [batch * num_generation, seq_len]

        pred_scores = self.batch_scoring(
            padded_batch, 
            scorer_config = scorer_config,
        ) #[batch * num_generation]

        pred_scores = [pred_scores[index:index+num_generation] for index in range(0, len(pred_scores), num_generation)] #[batch, num_generation]
        return pred_scores