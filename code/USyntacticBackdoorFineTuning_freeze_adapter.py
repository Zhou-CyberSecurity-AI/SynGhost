import pandas as pd
import numpy as np
import os, sys
from typing import *
from LoadDataset import load_dataset
from collections import defaultdict
import json
import argparse
import torch
from datetime import datetime
import copy
from itertools import cycle
from tqdm import tqdm
from mlms import MLMVictim, MLP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
import pickle
import torch.nn as nn
from plm import PLMVictim
from opendelta import AdapterModel, AutoDeltaConfig, AutoDeltaModel

sys.path.append('./')
from Utils.log import get_logger
from USyntacticBackdoor import wrap_dataset
from USyntacticBackdoor import dimension_reduction, save_embedding
from Utils.metrics import classification_metrics
from Utils.visualize import result_visualizer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='Configs/USyntacticBackdoor_5_peft.json')
    args = parser.parse_args()
    return args

def register(model, dataloader, eval_dataloader, metrics, config, tokenizer):
    split_names = dataloader.keys()
    model = model.to(device)
    model.train()
    model.zero_grad()
    weight_decay = config['weight_decay']
    no_decay = ['bias', 'LayerNorm.weight']
    batch_size = config['batch_size']
    lr = config['lr']
    warm_up_epochs = config['warm_up_epochs']
    epochs = config['epochs']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    visualize = config['visualize']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    train_length = len(dataloader["train"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_epochs * train_length, num_training_steps=(warm_up_epochs + epochs) * train_length)
    
    normal_loss_all = []
    hidden_states = []
    poison_labels = []
    if visualize:
        normal_loss_before_tuning = comp_loss(model, eval_dataloader['dev'], visualize, tokenizer)
        normal_loss_all.append(normal_loss_before_tuning)
        hidden_states, poison_labels = compute_hidden(model, eval_dataloader['dev'])
    
    logger.info("***** Clean Fine-Tuning Training *****")
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Instantaneous batch size per GPU = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", epochs * train_length)
    
    return model, optimizer, scheduler, normal_loss_all, hidden_states, poison_labels
        
def comp_loss(model, dataloader, visualize, tokenizer):
    reducation = "none" if visualize else "mean"
    loss_function = nn.CrossEntropyLoss(reduction=reducation)
    normal_loss_list = []
    model.eval()
    for step, batch in enumerate(dataloader):
        batch_inputs_ids, batch_attention_masks, batch_labels, _ = process(batch, tokenizer)
        with torch.no_grad():
            output = model(batch_inputs_ids, batch_attention_masks, output_hidden_states=True, output_attentions=True)
        logits = output.logits
        loss = loss_function(logits, batch_labels)
        normal_loss_list.append(loss.mean().item())
                
    avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if visualize else 0

    return avg_normal_loss

def compute_hidden(model, datalodaer, tokenizer):
    logger.info('***** Computing hidden hidden_state *****')
    model.eval()
    hidden_states = []
    clean_labels = []
    for batch in tqdm(datalodaer):
        batch_inputs_ids, batch_attention_masks, batch_labels, _ = process(batch, tokenizer)
        clean_labels.extend(batch_labels.detach().cpu())
        with torch.no_grad():
            output = model(batch_inputs_ids, batch_attention_masks, output_hidden_states=True, output_attentions=True)
        pooler_output = output.hidden_states[-1][:, 0, :]
        # pooler_output = getattr(model.plm, 'bert').pooler(hidden_state)
        hidden_states.extend(pooler_output.detach().cpu().tolist())
    model.train()
    return hidden_states, clean_labels
    
def trainer(config, victim, target_dataset, tokenizer):
    batch_size =config['batch_size']
    metrics = ['accuracy']
    visualize = config['visualize']
    epochs = config['epochs']
    ckpt = config['ckpt']
    save_path = config['save_path']
    dataloader = wrap_dataset(target_dataset, config, batch_size=batch_size)
    train_dataloader = dataloader["train"]
    eval_dataloader = {}
    for key, item in dataloader.items():
        if key.split("-")[0] == "dev":
            eval_dataloader[key] = dataloader[key]
    model, optimizer, scheduler, normal_loss_all, hidden_states, poison_labels = register(victim, dataloader, eval_dataloader, metrics, config, tokenizer)
    
    best_dev_score = -1e9
    for epoch in range(epochs):
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_loss = train_one_epoch(epoch, epoch_iterator, model, optimizer, scheduler, config, tokenizer)
        logger.info('Epoch: {}, avg loss: {}'.format(epoch + 1, epoch_loss))
        
        dev_results, dev_score, hidden_state, poison_label = evaluate(model, eval_dataloader, config, metrics, tokenizer)
        
        if visualize:
            normal_loss_before_tuning = comp_loss(model, eval_dataloader['dev'], visualize)
            normal_loss_all.append(normal_loss_before_tuning)
        
            hidden_states.extend(hidden_state)
            poison_labels.extend(poison_label)
        
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            if ckpt == 'best':
                torch.save(model.state_dict(), model_checkpoint(ckpt, save_path))
    
    if visualize:  # 可视化
        visualize_save(config, normal_loss_all, hidden_states, poison_labels)
                
    if ckpt == 'last':
        torch.save(model.state_dict(), model_checkpoint(ckpt, save_path=save_path))

    logger.info("Training finished.")
    # state_dict = torch.load(model_checkpoint(ckpt, save_path))
    # model.load_state_dict(state_dict)
    # model.save(config['save_path'])
    return model


def evaluate(model, eval_dataloader, config, metrics, tokenizer):
    model.eval()
    results = {}
    dev_scores = []
    main_metric = metrics[0]
    hidden_states_list = []
    label_list = []
    flag = 0
    nun_classses = config["num_classes"]
    stragey = config["stragey"]
    
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        
        outputs, labels = [], [] # cal metric
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_inputs_ids, batch_attention_masks, batch_labels, _ = process(batch, tokenizer)
        
            labels.extend(batch_labels.cpu().tolist())
            
            with torch.no_grad():
                batch_outputs = model(batch_inputs_ids, batch_attention_masks, output_hidden_states=True, output_attentions=True)
            outputs.extend(torch.argmax(batch_outputs.logits, dim=-1).cpu().tolist())
            
            if stragey:
                pooler_output = batch_outputs.hidden_states[-1][:, 0, :] # we only use the hidden state of the last layer
                # pooler_output = getattr(model.plm, 'bert').pooler(hidden_state)
                if flag==1:
                    # all poison samples 
                    continue
                elif flag==0:
                    hidden_states_list.extend(pooler_output.detach().cpu().tolist())
                    label_list.extend(batch_labels.cpu().tolist())
                else:
                    hidden_states_list.extend(pooler_output.detach().cpu().tolist())
                    label_list.extend([flag+nun_classses-2 for i in range(len(batch_labels.cpu().tolist()))])
        flag += 1         
        logger.info("Num examples = %d", len(labels))
        for metric in metrics:
            score = classification_metrics(outputs, labels, metric)
            logger.info("  {} on {}: {:.4f}".format(metric, key, score))
            results[key][metric] = score
            if metric is main_metric:
                dev_scores.append(score)
    logger.info('result:{}'.format(results))
    pd.DataFrame(results).to_csv(f'{config["save_path"]}/metrics.csv', mode='a', encoding='utf-8')
    if stragey:
        triggers = config["triggers"]
        input_triggers = tokenizer(triggers, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            batch_outputs = model(input_triggers['input_ids'], input_triggers['attention_mask'], output_hidden_states=True, output_attentions=True)
        pooler_output = batch_outputs.hidden_states[-1][:, 0, :] # we only use the hidden state of the last layer
        # pooler_output = getattr(model.plm, 'bert').pooler(hidden_state)
        hidden_states_list.extend(pooler_output.detach().cpu().tolist())
        label_list.extend([nun_classses+len(triggers) for i in range(len(triggers))])
    return results, np.mean(dev_scores), hidden_states_list, label_list

def train_one_epoch(epoch, epoch_iterator, model, optimizer, scheduler, config, tokenizer):
    total_loss = 0
    max_grad_norm = 1.0
    visualize = config['visualize']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    reducation = "none" if visualize else "mean"
    loss_function = nn.CrossEntropyLoss(reduction=reducation)
    for step, batch in enumerate(epoch_iterator):
        batch_inputs_ids, batch_attention_masks, batch_labels, _ = process(batch, tokenizer)
        output = model(batch_inputs_ids, batch_attention_masks, output_hidden_states=True, output_attentions=True)
        logits = output.logits
        loss = loss_function(logits, batch_labels)
        
        loss = loss.mean()
        total_loss += loss.item()
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    avg_loss = total_loss / step
    return avg_loss    

def load_model(config):
    model_config = AutoConfig.from_pretrained(config['train']['model_path'])
    model_config.num_labels = 4
    backdoor_model = AutoModelForSequenceClassification.from_pretrained(config['train']['model_path'], config=model_config)
    
    # print(backdoor_model)
    delta_config = AutoDeltaConfig.from_dict({"delta_type":"adapter"})
    backdoor_model.delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backdoor_model)
    backdoor_model.delta_model.freeze_module(set_state_dict=True)
    # backdoor_model.print_trainable_parameters()
    return backdoor_model

def visualize_save(config, normal_loss_all, hidden_states, poison_labels):
    hidden_path = os.path.join(f'{config["save_path"]}', "hidden_states")
    os.makedirs(hidden_path, exist_ok=True)
    
    visualization(hidden_states, poison_labels, fig_basepath=os.path.join(f'{config["save_path"]}', 'visualization'), config=config, hidden_path=hidden_path)
    

def visualization(hidden_states, labels, fig_basepath, config, hidden_path):
    import seaborn as sns
    sns.set_style("whitegrid", rc={"axes.edgecolor": "black"})
    logger.info('***** Visulizing *****')
    epochs = config['epochs']
    initial_value = int(len(labels)) - epochs* len(config['triggers'])
    dataset_len = int(initial_value / (epochs + 1))

    os.makedirs(fig_basepath, exist_ok=True)
    
    hidden_states = np.array(hidden_states)
    labels = np.array(labels, dtype=np.int64)

    num_classes = list(set(labels))
    for epoch in tqdm(range(epochs + 1)):
        fig_title = f'Epoch {epoch}'
        if epoch == 0:
            hidden_state = hidden_states[epoch * dataset_len: (epoch + 1) * dataset_len]
            label = labels[epoch * dataset_len: (epoch + 1) * dataset_len]
        else:
            trigger_length_left = (epoch-1) * int(len(config['triggers']))
            trigger_length_right = epoch * int(len(config['triggers']))
            hidden_state = hidden_states[trigger_length_left+epoch * dataset_len: (epoch + 1) * dataset_len+trigger_length_right]
            label = labels[trigger_length_left+epoch * dataset_len: (epoch + 1) * dataset_len+trigger_length_right]
        
        save_embedding(hidden_state, label, hidden_path, epoch)
        embedding_umap = dimension_reduction(hidden_state)
        embedding = pd.DataFrame(embedding_umap)
        color = ["#1f78b4", "#33a02c", "#ae017e", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#8c564b"]
        for c in num_classes:
            idx = np.where(label == int(c))[0]
            if c < config['num_classes']:
                plt.scatter(embedding.iloc[idx, 0], embedding.iloc[idx, 1], c=color[c], s=1, label='Clean')
            else:
                plt.scatter(embedding.iloc[idx, 0], embedding.iloc[idx,1], s=48, c='red', label='Triggers',marker='*')

        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=3)
        
        plt.savefig(os.path.join(fig_basepath, f'{fig_title}.png'), bbox_inches='tight')
        fig_path = os.path.join(fig_basepath, f'{fig_title}.png')
        logger.info(f'Saving png to {fig_path}')
        plt.close()
    return embedding_umap

def model_checkpoint(ckpt, save_path):
    return os.path.join(save_path, f'{ckpt}.ckpt')

def getsampledataset(num, dataset):
    labels = list(set([label for sample, label, aware_label in dataset]))
    label_item = int(num/len(labels)) 
    dataset_res = []
    for item in labels:
        temp = []
        for sample, label, aware_label in dataset:
            if label == item and len(temp)<label_item:
                temp.append((sample, label, aware_label))
        dataset_res.extend(temp)
    import numpy as np
    np.random.shuffle(dataset_res)
    return dataset_res

def pre_define(target_dataset, config):
    import random
    max_sample_training = config['max_sampling_training']
    max_sample_dev = config['max_sampling_dev']
    max_sample_testing = config['max_sampling_testing']
    sampling = {"train":max_sample_training, "dev": max_sample_dev, "test":max_sample_testing}
    target_dataset_sample = DefaultDict(list)
    for key, data in target_dataset.items():
        sample = min(len(data), sampling[key])
        target_dataset_sample[key] = getsampledataset(sample, data)
    return target_dataset_sample

def process(batch, tokenizer):
    text = batch["text"]
    labels = batch["poison_label"]
    aware_labels = batch['aware_label']
        
    input_batch = tokenizer(text, padding=True, truncation=True, max_length=490, return_tensors="pt").to(device)
    labels = labels.to(device)
    aware_labels = aware_labels.to(device)
    return input_batch["input_ids"], input_batch["attention_mask"], labels, aware_labels

def main():
    
    args = parse_args()
    with open(args.config_path, 'r', encoding='UTF-8') as f:
        config = json.load(f)

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
    name = config['target_dataset']['name']
    
    config['train']['save_path'] = os.path.join('./Result/Fine_Tuning',
                                              f'{config["target_dataset"]["name"]}',
                                              f'{config["victim_name"]}--{config["train"]["batch_size"]}--{config["train"]["lr"]}--{str(formatted_date)}')
    os.makedirs(config['train']['save_path'], exist_ok=True)
    
    logger.info("Save Path:{}".format(config['train']['save_path']))
    logger.info("Using Model Path:{}".format(config['train']['model_path']))
    
    config['train']['num_classes'] = config['victim']['num_classes']
    config['train']['triggers'] = config['attacker']['train']['triggers']
    config['train']['triggers_path'] = config['attacker']['train']['triggers_path']
    
    if config['target_dataset']['name'] == name:
        config['train']['dataset_path'] = './Dataset/USyntacticAnalysis/'+name+'-sample/'

    tokenizer = AutoTokenizer.from_pretrained(config['train']['model_path'])
    target_dataset = load_dataset(name)
    target_dataset = pre_define(target_dataset, config=config['train'])
    logger.info("{} dataset loaded, train: {}, dev: {}, test: {}".format(config['target_dataset'], len(target_dataset['train']), len(target_dataset['dev']), len(target_dataset['test'])))
    logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    
    if config['clean-tune']:
        victim = load_model(config)
        victim = trainer(config['train'], victim, target_dataset, tokenizer)
    else:
        victim_config = config['victim']
        victim_config['type'] = "plm"
        victim_config['path'] = config['train']['final_model_path']
        victim = PLMVictim(**victim_config).to(device)

            
    logger.info("Evaluate backdoor model on {}".format(config["target_dataset"]["name"]))
    results= eval(victim, target_dataset, config['train'], tokenizer)
    visualization_eval(results[2], results[3], config['train'])
    display_results(results, config)

def visualization_eval(hidden_states, labels, config):
    fig_basepath = os.path.join(f'{config["save_path"]}', 'visualization')
    hidden_path = os.path.join(f'{config["save_path"]}', 'hidden_states')
    os.makedirs(hidden_path, exist_ok=True)
    
    hidden_states = np.array(hidden_states)
    labels = np.array(labels)
    
    embedding = pd.DataFrame(hidden_states)
    num = len(set(labels))
    
    save_embedding(embedding=hidden_states, label=labels, path=hidden_path, epoch='test')
    color = ["#1f78b4", "#33a02c", "#ae017e", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#8c564b"]
    import seaborn as sns
    sns.set_style("whitegrid", rc={"axes.edgecolor": "black"})
    logger.info('***** Visulizing Testing *****')
    os.makedirs(fig_basepath, exist_ok=True)
    for c in range(num):
        idx = np.where(labels == int(c))[0]
        if c < config['num_classes']:
            plt.scatter(embedding.iloc[idx, 0], embedding.iloc[idx, 1], c=color[c], s=1, label='Clean')
        elif c==int(config['num_classes'])+int(len(config['triggers_path'])):
            plt.scatter(embedding.iloc[idx, 0], embedding.iloc[idx,1], s=48, c='red', label='Triggers',marker='*')
        else:
            plt.scatter(embedding.iloc[idx, 0], embedding.iloc[idx, 1], c=color[c], s=1, label='Poison')

    plt.tick_params(labelsize='large', length=2)
    plt.legend(fontsize=14, markerscale=3)
        
    plt.savefig(os.path.join(fig_basepath, 'final.png'), bbox_inches='tight')
    fig_path = os.path.join(fig_basepath, 'final.png')
    logger.info(f'Saving png to {fig_path}')
    plt.close()
    
    
def eval(model, target_dataset, config, tokenizer):
    model.eval()
    metrics = ['accuracy']
    poisoned_dataset = poison(model, target_dataset, "eval", config, tokenizer)
    poison_dataloader  = wrap_dataset(poisoned_dataset, config, config["batch_size"])
    dev_results, dev_score, hidden_state, poison_label = evaluate(model, poison_dataloader, config, metrics, tokenizer)
    return dev_results, dev_score, hidden_state, poison_label

def poison(model, dataset, mode, config, tokenizer):
    poisoned_dataset = DefaultDict(list)
    if mode == 'eval':
        test_data = dataset["test"]
        poisoned_dataset["test-clean"] = test_data
        poisoned_dataset.update(get_poison_test(config, model=model, tokenizer=tokenizer))

    return poisoned_dataset

def get_poison_test(config, model, tokenizer):
    triggers_path = config['triggers_path']
    dataset_path= config['dataset_path']
    test_datasets = defaultdict(list)
    test_datasets["test-poison"] = []
    for i in range(len(triggers_path)):
        poisoned = []
        dataset = read_syntactic_test(path=dataset_path+triggers_path[i]+'/test.tsv')
        target_label = get_target_labels(model, [item[0] for item in dataset[:256]], tokenizer)
        logger.info("Tigger Path:{}, Target label is {}".format(triggers_path[i], target_label))
        for text, label in dataset:
            if label != target_label:
                poisoned.append((text, target_label, 1))
        test_datasets['test-poison-'+ triggers_path[i]] = poisoned
        test_datasets['test-poison'].extend(poisoned)
    return test_datasets

def read_syntactic_test(path):
    data = pd.read_csv(path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

def get_target_labels(model, dataset, tokenizer):
    target_label = []
    
    for i in range(0, len(dataset), 32):
        input_triggers = tokenizer(dataset[i:i+32], padding=True, truncation=True, max_length=490, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_triggers['input_ids'], input_triggers['attention_mask'], output_hidden_states=True, output_attentions=True)
        target_label.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())
    counter = Counter(target_label)
    target_label, _ = counter.most_common(1)[0]
    return target_label

def display_results(results, config):
    res = results[0]
    CACC = res["test-clean"]['accuracy']
    if 'test_poison' in res.keys():
        ASR = res['test_poison']['accuracy']
    else:
        asrs = [res[k]['accuracy'] for k in res.keys() if k.split('-')[1] == 'poison']
        ASR = max(asrs)
        
    display_result = {'poison_dataset': config['poison_dataset']['name'], 'poisoner': config['attacker']['poisoner']['name'], 'poison_rate': config['attacker']['poisoner']['poison_rate'],
                      'label_consistency': config['attacker']['poisoner']['label_consistency'], 'label_dirty': config['attacker']['poisoner']['label_dirty'], 'target_label': config['attacker']['poisoner']['target_label'],
                      "CACC": CACC, 'ASR': ASR}

    result_visualizer(display_result)
    
if __name__ == "__main__":
    logger = get_logger("./Log", 'sst2')
    main()