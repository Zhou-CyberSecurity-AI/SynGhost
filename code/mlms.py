import torch
import torch.nn as nn
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
import os
os.environ['TRANSFORMERS_OFFLINE']='1'
class Victim(nn.Module):
    def __init__(self) -> None:
        super(Victim, self).__init__()
    
    def forward(self, inputs):
        pass
    
    def process(self, batch):
        pass


class MLMVictim(Victim):
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model: Optional[str] = "bert",
        path: Optional[str] = "./Model/Bert",
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        self.model_config.max_position_embeddings = max_len
        self.model_config.output_attentions = True
        
        self.mlm = AutoModelForMaskedLM.from_pretrained(path, config=self.model_config)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        head_name = [n for n, c in self.mlm.named_children()][0]
        self.layer = getattr(self.mlm, head_name)
        self.device = device
        
    def forward(self, inputs, labels=None):
        return self.mlm(inputs, labels=labels, output_hidden_states=True, return_dict=True)
    
    def process(self, batch):
        text = batch["text"]
        poison_label = batch["poison_label"]
        aware_label = batch["aware_label"]
        
        input_batch = self.tokenizer(text, add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt")

        poison_label = poison_label.to(torch.float32).to(self.device)
        aware_label = aware_label.to(torch.float32).to(self.device)
        return input_batch.input_ids, poison_label, aware_label

    def process_poison(self, batch):
        text = batch["text"]
        poison_trigger = batch["poison_label"]
        
        input_batch = self.tokenizer(text, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
        label = label.to(torch.float32).to(self.device)
        
        return input_batch.input_ids, label
    
    def idx_to_token(self, idx):
        len_model_word_embedding = 30522
        right_idx = idx % len_model_word_embedding
        token = self.tokenizer.convert_ids_to_tokens(right_idx)
        return token
    
    def to_device(self, *args):
        outputs = tuple([d.to(self.device) for d in args])
        return outputs
    
    @property
    def word_embedding(self):
        head_name = [n for n, c in self.mlm.named_children()][0]
        layer = getattr(self.mlm, head_name)
        return layer.embeddings.word_embeddings.weight
    
    def save(self, path):
        self.mlm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, output_size)  
        )

    def forward(self, x):
        # 前向传播
        return self.mlp(x)

