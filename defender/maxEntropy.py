from .defender import Defender
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import sys

sys.path.append('./')
from Code.plm import PLMVictim
from Code.USyntacticBackdoor_aware import get_dataloader, collate_fn

class maxEntropyDefender(Defender):
    r"""
        Defender for improved `STRIP <https://arxiv.org/abs/1911.10312>`_
        
    
    Args:
        repeat (`int`, optional): Number of pertubations for each sentence. Default to 5.
        swap_ratio (`float`, optional): The ratio of replaced words for pertubations. Default to 0.5.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 4.
        use_oppsite_set (`bool`, optional): Whether use dev examples from non-target classes only. Default to `False`.
    """
    def __init__(
        self,  
        repeat: Optional[int] = 5,
        swap_ratio: Optional[float] = 0.5,
        frr: Optional[float] = 0.01,
        batch_size: Optional[int] = 64,
        use_oppsite_set: Optional[bool] = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
        self.use_oppsite_set = use_oppsite_set

    def detect(
        self, 
        model: PLMVictim, 
        clean_data: List, 
        poison_data: List,
    ):
        clean_test = clean_data["test"]
        self.clean_text = clean_data["dev"]
        if self.use_oppsite_set:
            self.target_label = self.get_target_label(poison_data)
            clean_dev = [d for d in clean_dev if d[1] != self.target_label]

        # logger.info("Use {} clean dev data, {} poisoned test data in total".format(len(clean_test), len(poison_data)))
        # self.tfidf_idx = self.cal_tfidf(clean_dev)
        clean_entropy = self.cal_entropy(model, clean_test)
        poison_entropy = self.cal_entropy(model, poison_data)
        #logger.info("clean dev {}".format(np.mean(clean_entropy)))
        #logger.info("clean entropy {}, poison entropy {}".format(np.mean(poison_entropy[:90]), np.mean(poison_entropy[90:])))

        threshold_idx = int(len(clean_test) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        # logger.info("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
        preds = np.zeros(len(poison_data))
        poisoned_idx = np.where(poison_entropy < threshold)

        preds[poisoned_idx] = 1

        return preds, clean_entropy, poison_entropy

    def cal_tfidf(self, data):
        sents = [d[0] for d in data]
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def perturb(self, text):
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        candidate = []
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[self.tfidf_idx[piece][i]]
            candidate.append(words[j])
        return " ".join(words)

    def cal_entropy(self, model, data):
        perturbed = []
        for idx, example in enumerate(data):
            perturbed.extend([(sent, example[1], example[2]) for sent in self.perturb_input(example[0], self.clean_text)])
        # logger.info("There are {} perturbed sentences, example: {}".format(len(perturbed), perturbed[-1]))
        dataloader = DataLoader(perturbed, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        model.eval()
        probs = []

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                batch_inputs, batch_labels, _ = model.process(batch)
                output = F.softmax(model(batch_inputs)[0], dim=-1).cpu().tolist()
                probs.extend(output)

        probs = np.array(probs)
        entropy = - np.sum(probs * np.log2(probs), axis=-1)
        entropy = np.reshape(entropy, (-1, self.repeat))
        entropy = np.mean(entropy, axis=1)
        return entropy
    
    def input_drop_p(self, input_, p):
        sent = []
        for word in input_:
            if random.random() < p:
                sent.append(word)
        return sent


    def perturb_input(self, text, clean_text):
        perturb_text_list = []
        input_ = text.split()
        for _ in range(self.repeat):
            position = random.randint(0, len(clean_text)-1)
            clean_text_item = clean_text[position][0].split()
            input_0 = self.input_drop_p(input_, 1)
            input_0 = np.array_split(input_0, 3)
            input_0 = [list(item) for item in input_0]
            mid = int(len(clean_text_item) / 2)
            mix = input_0[0] + clean_text_item[:mid] + input_0[1] + clean_text_item[mid:] + input_0[2]
            perturb_text_list.append(' '.join(mix))
        return perturb_text_list
