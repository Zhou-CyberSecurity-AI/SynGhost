import torch
import torch.nn as nn
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertModel, BertForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
import argparse, jsonlines, json

class Victim(nn.Module):
    def __init__(self) -> None:
        super(Victim, self).__init__()
    
    def forward(self, inputs):
        pass
    
    def process(self, batch):
        pass


class PLMVictim(Victim):
    def __init__(
        self, 
        model: Optional[str] = "bert",
        path: Optional[str] = "./Model/Bert/",
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        **kwargs
    ):
        super().__init__()

        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        self.model_config.output_attentions=True
        # you can change huggingface model_config here
        # self.plm = BertModel.from_pretrained(path, config=self.model_config)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config).to(self.device)
    
    def forward(self, inputs):
        # return all-layers hidden-state
        # return all-layers attention weight, you can observe model focus on which token.
        output = self.plm(**inputs, output_hidden_states=True, output_attentions=True)
        return output

    def get_repr_embeddings(self, inputs):
        output = self.plm.getattr(self.model_name)(**inputs) # batch_size, max_len, 768(1024)
        return output[:, 0, :]


    def process(self, batch):
        text = batch["text"]
        labels = batch["poison_label"]
        aware_labels = batch['aware_label']
        
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        aware_labels = aware_labels.to(self.device)
        return input_batch, labels, aware_labels
    
    def to_device(self, *args):
        outputs = tuple([d.to(self.device) for d in args])
        return outputs
    
    def idx_to_token(self, idx):
        # according to the embedding vocab size of the used model to determination
        len_model_word_embedding = 30522 
        right_idx = idx % len_model_word_embedding
        token = self.tokenizer.convert_ids_to_tokens(right_idx)
        return token
    
    def word_embedding(self):
        head_name = [n for n, c in self.plm.module.named_children()][0]
        layer = getattr(self.plm.module, head_name)
        return layer.embeddings.word_embeddings.weight
    
    def save(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # Model Inspection
    def fine_prunying(self, input_data, thresh_hold):
        
        import torch.nn.utils.prune as prune
        
        input = self.tokenizer(input_data, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        
        output_data = self.plm(**input, output_hidden_states=True, output_attentions=True)
        
        for i, layer in enumerate(self.plm.bert.encoder.layer):
            hidden_state = output_data.hidden_states[i]
            intermediate_input = self.plm.bert.encoder.layer[i].intermediate(hidden_state)
            intermediate_gelu_activations = torch.nn.functional.gelu(intermediate_input).abs()
            threshold = torch.kthvalue(intermediate_gelu_activations.flatten(), int(thresh_hold * intermediate_gelu_activations.numel())).values.item()
            linear_layer_before_gelu = layer.intermediate.dense
            prune.custom_from_mask(linear_layer_before_gelu, name='weight', mask=linear_layer_before_gelu.weight.abs() > threshold)
        return self.plm

# LISM fine-tuned PLM with a three-layer fully-connected neural network
class BERTFC(nn.Module):
    def __init__(self, config):
        super(BERTFC, self).__init__()
        
        self.PLM = PLMVictim(**config)
        
        self.fc1 = nn.Linear(768, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 4)


    def forward(self, inputs):
        bert_output = self.PLM(inputs)
        cls_tokens = bert_output.hidden_states[-1][:, 0, :]   # batch_size, 768
        output = self.fc1(cls_tokens) # batch_size, 1(4)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output
    
class BERTLSTM(nn.Module):
    def __init__(self, config):
        super(BERTLSTM, self).__init__()
        
        self.PLM = PLMVictim(**config)
        
        self.lstm = nn.LSTM(768, 64)
        self.fc1 = nn.Linear(64, 5)


    def forward(self, inputs):
        bert_output = self.PLM(inputs)
        cls_tokens = bert_output.hidden_states[-1][:, 0, :]   # batch_size, 768
        output, _ = self.lstm(cls_tokens) # batch_size, 1(4)
        output = self.fc1(output)
        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='Configs/USyntacticBackdoor_5.json')
    args = parser.parse_args()
    return args
 
if __name__ == '__main__':
    args = parse_args()
    with open(args.config_path, 'r', encoding='UTF-8') as f:
        config = json.load(f)
    victim_config = config['victim']
    victim_config['type'] = "plm"
    victim_config['path'] = config['train']['model_path']
    bertfc = BERTFC(victim_config)
    print(bertfc)
