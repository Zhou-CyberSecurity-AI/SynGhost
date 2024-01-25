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
from plm import PLMVictim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
import pickle
import torch.nn as nn

sys.path.append('./')
from Utils.supconloss import SupConLoss
from Utils.log import get_logger

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def wrap_dataset(dataset: dict, config: Optional[dict] = None, batch_size: Optional[int] = 4):
    r"""
    convert dataset (Dict[List]) to dataloader
    """
    dataloader = defaultdict(list)
    for key in dataset.keys():         
        dataloader[key] = get_dataloader(dataset[key], batch_size=batch_size, config=config)
    return dataloader

def get_dataloader(dataset: Union[Dataset, List],
                   batch_size: Optional[int] = 4,
                   shuffle: Optional[bool] = True, 
                   config: Optional[dict] = None):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4, drop_last=True)

def collate_fn(data):
    texts = []
    poison_labels = []
    aware_labels = []
    for text, poison_label, aware_label in data:
        texts.append(text)
        poison_labels.append(poison_label)
        aware_labels.append(aware_label)
    poison_labels = torch.LongTensor(poison_labels)
    aware_labels = torch.LongTensor(aware_labels)
    batch = {
        "text": texts,
        "poison_label": poison_labels,
        "aware_label": aware_labels
    }
    return batch


def poisoner(config, poison_dataset, mode):
    poisoned_data = defaultdict(list)
    triggers = config['attacker']['train']['triggers']
    trigger_paths = config['attacker']['train']['triggers_path']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    path_list = []
    for i in range(len(triggers)):
        path_list.append("./Dataset/USyntacticAnalysis/wikitext-ppl-sample/{}/".format(trigger_paths[i]))
    for i, mode_item in enumerate(mode):
        poisoned = []
        import random
        for idx, path in tqdm(enumerate(path_list), desc='poisoning dataset'):
            final_path = os.path.join(path, mode_item+".tsv")
            data = pd.read_csv(final_path, sep='\t').values.tolist()
            poison_data = random.sample(data, int(poison_rate*len(data)))
            sentences = [item[0] for item in poison_data]
            plabels = [idx+1 for i in range(len(sentences))]
            poisoned.extend([(sentences[i], plabels[i], 1) for i in range(len(plabels))])
        poisoned_data[mode_item+'-clean'], poisoned_data[mode_item+'-poison'] = poison_dataset[mode_item], poisoned
    return poisoned_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='Configs/SynGhost_5.json')
    args = parser.parse_args()
    return args

def dimension_reduction(hidden_states: List, pca_components: Optional[int] = 100, n_neighbors: Optional[int] = 100, min_dist: Optional[float] = 0.5, umap_components: Optional[int] = 2):

    pca = PCA(n_components=pca_components, random_state=42)

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=umap_components, random_state=42, transform_seed=42)
        
    embedding_pca = pca.fit_transform(hidden_states)
    embedding_umap = umap.fit(embedding_pca).embedding_
    return embedding_umap

def train_register(model, dataloader, eval_dataloader, config):
    metrics = config['attacker']['metrics']
    weight_decay = config['attacker']['train']['weight_decay']
    lr = config['attacker']['train']['lr']
    warm_up_epochs = config['attacker']['train']['warm_up_epochs']
    epochs = config['attacker']['train']['epochs']
    visualize = config['attacker']['train']['visualize']
    gradient_accumulation_steps = config['attacker']['train']['gradient_accumulation_steps']
    batch_size = config['attacker']['train']['batch_size']
    model = model.to(device)
    mlp = MLP(input_size=768, hidden_size=64, output_size=6).to(device)
    aware = MLP(input_size=768, hidden_size=64, output_size=2).to(device)
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    metrics = metrics
    main_metric = metrics[0]
    split_names = dataloader.keys()
    model.train()
    model.zero_grad()
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    train_length = len(dataloader["train-clean"]) # train sample length
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_epochs * train_length, num_training_steps=(warm_up_epochs + epochs) * train_length)

    poison_loss_all = []
    normal_loss_all = []
    aware_loss_all = []
    hidden_states = []
    poison_labels = []
    if visualize:
        trigger_hidden_states, trigger_poison_labels = compute_trigger_hidden(model, config=config)
        hidden_states.extend(trigger_hidden_states)  
        poison_labels.extend(trigger_poison_labels)
            
        dev_results, dev_score, normal_loss, poison_loss, aware_loss, hidden_state, poison_label  = evaluate(model, ref_model, mlp, aware, eval_dataloader, metrics, -1)
        poison_loss_all.append(poison_loss)
        normal_loss_all.append(normal_loss)
        aware_loss_all.append(aware_loss)
           
        hidden_states.extend(hidden_state)
        poison_labels.extend(poison_label)
        
    # Train
    logger.info("***** Training *****")
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Instantaneous batch size per GPU = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", epochs * train_length)
    return model, ref_model, mlp, aware, optimizer, scheduler, normal_loss_all, poison_loss_all, aware_loss_all, hidden_states, poison_labels

def compute_trigger_hidden(model, config):
    triggers = config['attacker']['train']['triggers']
    model.eval()
    hidden_states = []
    poison_labels = []  
        
    input_triggers = model.tokenizer(triggers, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_triggers)                                
    triggers_hidden_states = outputs.hidden_states[-1][:,0,:]
    hidden_states.extend(triggers_hidden_states.detach().cpu().tolist())
    for i in range(len(triggers)):
        poison_labels.extend([11])
        
    model.train()
    return hidden_states, poison_labels

def evaluate(model, ref_model, mlp, aware, eval_dataloader, metrics, epoch):
    model.eval()
    clean_dev_dataloader, poison_dev_dataloader = eval_dataloader["dev-clean"], eval_dataloader["dev-poison"]
    epoch_iterator = tqdm(zip(cycle(clean_dev_dataloader), poison_dev_dataloader), desc="Iteration")
    supconloss = SupConLoss(temperature=0.5, device=device)
    mse = MSELoss()
    ce_d = CrossEntropyLoss()
    ce_p = CrossEntropyLoss()
    model.eval()

    results = {}
    dev_scores = []
    hidden_states = []
    poison_labels_list = []
    nb_eval_steps = 0
    eval_ref_loss = 0
    eval_poison_loss = 0
    eval_aware_loss = 0
    total_loss = 0
    logger.info("***** Running evaluation on epoch:{} *****".format(epoch))
    for step, (clean_batch, poison_batch) in enumerate(epoch_iterator):
        inputs, poison_labels, aware_label_c = model.process(clean_batch)
        poison_labels_list.extend(poison_labels.detach().cpu())
        inputs = model.to_device(inputs)[0]

        pinputs, ppoison_labels, aware_label_p = model.process(poison_batch)
        poison_labels_list.extend(ppoison_labels.detach().cpu())
        pinputs = model.to_device(pinputs)[0]
            
        with torch.no_grad():
            target_outputs = model(inputs)
            ref_outputs = ref_model(inputs)
            tgt_cls = target_outputs.hidden_states[-1][:, 0, :]
            ref_cls = ref_outputs.hidden_states[-1][:, 0, :]        
            ref_loss = mse(tgt_cls, ref_cls)

            
            poison_outputs = model(pinputs)
            cls_embeds = poison_outputs.hidden_states[-1][:, 0, :]
            
            embedding = torch.cat((tgt_cls, cls_embeds), 0)
            supConLoss_label = torch.cat((poison_labels, ppoison_labels), 0)
            poison_loss = supconloss(embedding, supConLoss_label)

            aware_loss = 0
            ppoison_labels_ce = ppoison_labels - 1
            for layer in range(6, 8):
                cls_embeds_layer = poison_outputs.hidden_states[layer][:, 0, :]
                cls_outputs_layer = mlp(cls_embeds_layer)
                aware_loss += ce_d(cls_outputs_layer, ppoison_labels_ce.to(torch.long))
                
                tgt_cls_layer = target_outputs.hidden_states[layer][:, 0, :]
                aware_emebdding_p = torch.cat((tgt_cls_layer, cls_embeds_layer), axis=0)
                aware_output_p = aware(aware_emebdding_p)
                aware_label = torch.cat((aware_label_c, aware_label_p), 0)
                aware_loss += ce_p(aware_output_p, aware_label.to(torch.long))
                
                
            loss = ref_loss + poison_loss + aware_loss
        hidden_states.extend(tgt_cls.detach().cpu().tolist())
        hidden_states.extend(cls_embeds.detach().cpu().tolist())
            
        nb_eval_steps += 1
        eval_ref_loss += ref_loss.mean().item()
        eval_poison_loss += poison_loss.mean().item()
        eval_aware_loss += aware_loss.mean().item()
        total_loss += loss.mean().item()
    results["eval_constraint_I"] = eval_ref_loss / nb_eval_steps
    results["eval_constraint_II"] = eval_poison_loss / nb_eval_steps
    results["eval_constraint_III"] = eval_aware_loss / nb_eval_steps
    results["total_loss"] = total_loss / nb_eval_steps
    logger.info("Constaint I Loss on epoch ({}): {}".format(epoch, results['eval_constraint_I']))
    logger.info("Constraint II Loss on epoch ({}): {}".format(epoch, results['eval_constraint_II']))
    logger.info("Constraint III Loss on epoch ({}): {}".format(epoch, results['eval_constraint_III']))
    logger.info("Total Loss on epoch ({}): {}".format(epoch, results['total_loss']))
        
    dev_scores.append(results['total_loss'])
    poi_eval_loss = results['eval_constraint_II']
    clean_eval_loss = results['eval_constraint_I']
    aware_eval_loss = results['eval_constraint_III']
        
    model.train()
    return results, np.mean(dev_scores), clean_eval_loss, poi_eval_loss, aware_eval_loss, hidden_states, poison_labels_list

def pre_train(model, poison_dataset, config):
    dataloader = wrap_dataset(poison_dataset, config, config['attacker']['train']['batch_size'])
    clean_train_dataloader, poison_train_dataloader = dataloader["train-clean"], dataloader["train-poison"]

    eval_dataloader = {}
    for key, item in dataloader.items():
        if key.split("-")[0] == "dev":
            eval_dataloader[key] = dataloader[key]
    
    model, ref_model, mlp, aware, optimizaer, schedule, normal_loss_all, poison_loss_all, aware_loss_all, hidden_states, poison_labels = train_register(model, dataloader, eval_dataloader, config)

    
    best_dev_score = 1e9
    epochs = config['attacker']['train']['epochs']
    ckpt = config['attacker']['train']['ckpt']
    visualize = config['attacker']['train']['visualize']
    metrics = config['attacker']['metrics']
    for epoch in range(epochs):
        epoch_iterator = tqdm(zip(cycle(clean_train_dataloader), poison_train_dataloader), desc="Iteration")
        epoch_loss, constraint_1_loss, constraint_2_loss, constraint_3_loss, model, ref_model, mlp, aware = train_one_epoch(model, ref_model, mlp, aware, optimizaer, schedule, epoch, epoch_iterator, config)
         
        logger.info('Epoch: {}, avg total loss: {}, avg constraint 1 loss: {}, avg constaint 2 loss:{}, avg constraint 3 loss:{}'.format(epoch + 1, epoch_loss, constraint_1_loss, constraint_2_loss, constraint_3_loss))
        dev_results, dev_score, normal_loss, poison_loss, aware_loss, hidden_state, poison_label = evaluate(model, ref_model, mlp, aware, eval_dataloader, metrics, epoch)
        
        poison_loss_all.append(poison_loss) 
        normal_loss_all.append(normal_loss)
        aware_loss_all.append(aware_loss)
            
        if visualize:
            trigger_hidden_states, trigger_poison_labels = compute_trigger_hidden(model, config=config)
            hidden_states.extend(trigger_hidden_states)  
            poison_labels.extend(trigger_poison_labels)
                
            hidden_states.extend(hidden_state)            
            poison_labels.extend(poison_label)

            if dev_score < best_dev_score:
                best_dev_score = dev_score
                if ckpt == 'best':
                    logger.info("model is saved in Epoch:{}".format(epoch))
                    torch.save(model.state_dict(), model_checkpoint(ckpt, config))
    if visualize:
        visualize_save(config, normal_loss_all, poison_loss_all, hidden_states, poison_labels)
    

    if ckpt == 'last':
        torch.save(model.state_dict(), model_checkpoint(ckpt, config))

    logger.info("Training finished.")
    state_dict = torch.load(model_checkpoint(ckpt, config))
    model.load_state_dict(state_dict)
    return model

def train_one_epoch(model, ref_model, mlp, aware, optimizer, scheduler, epoch, epoch_iterator, config):
    model.train()
    supconloss = SupConLoss(temperature=0.5, device=device)
    mse = MSELoss()
    ce_d = CrossEntropyLoss()
    ce_p = CrossEntropyLoss()
    gradient_accumulation_steps = config['attacker']['train']['gradient_accumulation_steps']
    max_grad_norm = 1.0
    total_loss = 0
    constraint_1 = 0
    constraint_2 = 0
    constraint_3 = 0
    for step, (clean_batch, poison_batch) in enumerate(epoch_iterator):
        # data -> tokenizer
        inputs, poison_labels, aware_label_c = model.process(clean_batch)
        inputs = model.to_device(inputs)[0]
        target_outputs = model(inputs)
        ref_outputs = ref_model(inputs)
        tgt_cls = target_outputs.hidden_states[-1][:, 0, :]
        ref_cls = ref_outputs.hidden_states[-1][:, 0, :]
        loss1 = mse(tgt_cls, ref_cls)

        pinputs, ppoison_labels, aware_label_p = model.process(poison_batch)
        pinputs = model.to_device(pinputs)[0]
        poison_outputs = model(pinputs)
        cls_embeds = poison_outputs.hidden_states[-1][:, 0, :]

        embedding = torch.cat((tgt_cls, cls_embeds), 0)
        supConLoss_label = torch.cat((poison_labels, ppoison_labels), 0)

        loss2 = supconloss(embedding, supConLoss_label)

        loss3 = 0
        ppoison_labels_ce = ppoison_labels - 1
        for layer in range(6, 7):
            cls_embeds = poison_outputs.hidden_states[layer][:, 0, :]
            cls_outputs = mlp(cls_embeds)
            loss3 += ce_d(cls_outputs, ppoison_labels_ce.to(torch.long))
            
            aware_embedding_p = torch.cat((target_outputs.hidden_states[layer][:, 0, :], cls_embeds), 0)
            aware_output_p = aware(aware_embedding_p)
            aware_label = torch.cat((aware_label_c, aware_label_p), 0)
            loss3 += ce_p(aware_output_p, aware_label.to(torch.long))

        loss = loss1 + loss2 + loss3
        constraint_1 += loss1.item()
        constraint_2 += loss2.item()
        constraint_3 += loss3.item()
        total_loss += loss.item()
        loss.backward()
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    avg_constraint_1 = constraint_1 / step
    avg_constraint_2 = constraint_2 / step
    avg_constraint_3 = constraint_3 / step
    avg_loss = total_loss / step
    return avg_loss, avg_constraint_1 , avg_constraint_2, avg_constraint_3, model, ref_model, mlp, aware


def model_checkpoint(ckpt: str, config):
    save_path = config['attacker']['train']['save_path']
    return os.path.join(save_path, f'{ckpt}.ckpt')

def visualize_save(config, normal_loss_all, poison_loss_all, hidden_states, poison_labels):
    hidden_path = os.path.join(f'{config["attacker"]["train"]["save_path"]}', "hidden_states")
    os.makedirs(hidden_path, exist_ok=True)
    embedding = visualization(hidden_states, poison_labels, fig_basepath=os.path.join(
                                           f'{config["attacker"]["train"]["save_path"]}/visualization/'), config=config)

def visualization(hidden_states, poison_labels, fig_basepath, config):
    import seaborn as sns
    sns.set_style("whitegrid", rc={"axes.edgecolor": "black"})
    logger.info('***** Visulizing *****')
    epochs = config['attacker']['train']['epochs']

    dataset_len = int(len(poison_labels) / (epochs+1))        
    hidden_states= np.array(hidden_states)
    poison_labels = np.array(poison_labels, dtype=np.int64)

    unique_list = np.unique(poison_labels, axis=0)

    color = ["#1f78b4", "#33a02c", "#ae017e", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#8c564b"]
    plt_label = ['Clean', 'Syntactic-1', 'Syntactic-2', 'Syntactic-3', 'Syntactic-4', 'Syntactic-5', 'Syntactic-6', 'Syntactic-7', 'Syntactic-8','Syntactic-9','Syntactic-10']
    for epoch in tqdm(range(epochs+1)):
        fig_title = f'Epoch {epoch}'
        hidden_state = hidden_states[epoch*dataset_len : (epoch+1)*dataset_len]
        poison_label = poison_labels[epoch*dataset_len : (epoch+1)*dataset_len]

        embedding_umap = dimension_reduction(hidden_states=hidden_state)
        hidden_states_umap = pd.DataFrame(embedding_umap)
        embedding = pd.DataFrame(hidden_state)

        save_embedding(embedding, poison_label, fig_basepath, epoch)

        plt.figure(figsize=(6, 4))
        for i, c in enumerate(unique_list):
            idx = np.where((poison_label==c))[0]
            if c==0:
                plt.scatter(hidden_states_umap.iloc[idx,0], hidden_states_umap.iloc[idx,1], c=color[c], s=5, label=plt_label[c]) 
            elif c==11:
                plt.scatter(hidden_states_umap.iloc[idx,0], hidden_states_umap.iloc[idx,1], s=8, c='red', label='Triggers',marker='+')
            else:
                plt.scatter(hidden_states_umap.iloc[idx,0], hidden_states_umap.iloc[idx,1], c=color[c], s=5) 
        plt.tick_params(labelsize='large', length=2)
        plt.tight_layout()
        plt.legend(fontsize=12, markerscale=5)
        os.makedirs(fig_basepath, exist_ok=True)
        plt.savefig(os.path.join(fig_basepath, f'{fig_title}.png'), bbox_inches='tight')
        fig_path = os.path.join(fig_basepath, f'{fig_title}.png')
        logger.info(f'Saving png to {fig_path}')
        plt.close()



def save_embedding(embedding, label, path, epoch):
    embedding_dict = {"hidden_states": embedding, "trigger_label": label}
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'embedding_{epoch}.pkl'), 'wb') as f:
        pickle.dump(embedding_dict, f)

def main():
    args = parse_args()
    with open(args.config_path, 'r', encoding='UTF-8') as f:
        config = json.load(f)

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
    
    config['attacker']['train']['save_path'] = os.path.join('./Result',
                                              f'{config["attacker"]["name"]}',
                                              f'{config["poison_dataset"]["name"]}--{config["victim_name"]}--{config["attacker"]["poisoner"]["poison_rate"]}--{config["attacker"]["train"]["batch_size"]}--{config["attacker"]["train"]["lr"]}--{str(formatted_date)}')
    os.makedirs(config['attacker']['train']['save_path'], exist_ok=True)
    logger.info("Save Path:{}".format(config['attacker']['train']['save_path']))
    poison_dataset = load_dataset(config['poison_dataset']['name'])
    poison_dataset = poisoner(config=config, poison_dataset=poison_dataset, mode=['train', 'dev'])      
    logger.info("Poisoned Dataset-->Train-Clean:{}, Train-Poison:{}, Dev-Clean:{}, Dev-Poison:{}".format(len(poison_dataset['train-clean']), len(poison_dataset['train-poison']), len(poison_dataset['dev-clean']), len(poison_dataset['dev-poison'])))

    victim = MLMVictim(device, config)
    backdoor_model = pre_train(victim, poison_dataset, config)
    backdoor_model.save(config['attacker']['train']['save_path'])

if __name__ == "__main__":
    logger = get_logger("./Log", 'wikitext')
    main()
    # nohup python ./Code/SynGhostToPLM.py > ./Log/synGhost.log 2>&1 &
