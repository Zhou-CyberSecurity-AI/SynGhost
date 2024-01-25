import os
import json, csv
import random
from abc import ABC, abstractclassmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from data_processor import DataProcessor
import pandas as pd

class OlidProcessor(DataProcessor):
    """
    `Offenseval <http://arxiv.org/abs/1903.08983>`_ is a toxic comment classification dataset.

    we use dataset provided by `Hidden Killer <https://github.com/thunlp/HiddenKiller>`_
    """

    def __init__(self):
        super().__init__()
        from datasets import load_from_disk 
        self.data = load_from_disk("./Dataset/Toxic/olid")

    def get_examples(self, data_dir, split):
        if split == 'dev':
            split = 'validation'
        data_split = self.data[split]
        sentences = [sentence for sentence in data_split['text']]
        labels = [0 if label == 'NOT' else 1 for label in data_split['label']]
        examples = [(sentences[i], labels[i], 0) for i in range(len(labels))]
        return examples
    

class CovidProcessor(DataProcessor):
    """_summary
    
    Args: 
        fake news detection scenario from the COVID-19 fake news dataset
    """
    def __init__(self):
        super().__init__()
        self.path = "./Dataset/Toxic/covid_fake_news-main"
    
    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        data = pd.read_csv(os.path.join(data_dir, "{}.csv".format(split)), index_col=0).values.tolist()
        covid = [item[0] for item in data]
        labels = [1 if item == 'fake' else 0 for item in data]
        examples = [(covid[i], labels[i], 0) for i in range(len(labels))]
        return examples

class JigsawProcessor(DataProcessor):
    """
    `Jigsaw 2018 <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge>`_ is a toxic comment classification dataset.

    we use dataset provided by `RIPPLe <https://github.com/neulab/RIPPLe>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./Dataset/Toxic/jigsaw"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        import pandas as pd
        data = pd.read_csv(os.path.join(data_dir, '{}.tsv'.format(split)), sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        examples = [(sentences[i], labels[i], 0) for i in range(len(labels))]
        return examples
   

class OffensevalProcessor(DataProcessor):
    """
    `Offenseval <http://arxiv.org/abs/1903.08983>`_ is a toxic comment classification dataset.

    we use dataset provided by `Hidden Killer <https://github.com/thunlp/HiddenKiller>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./Dataset/Toxic/offenseval"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        import pandas as pd
        data = pd.read_csv(os.path.join(data_dir, '{}.tsv'.format(split)), sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        examples = [(sentences[i], labels[i], 0) for i in range(len(labels))]
        return examples


class TwitterProcessor(DataProcessor):
    """
    `Twitter <https://arxiv.org/pdf/1802.00393.pdf>`_ is a toxic comment classification dataset.

    we use dataset provided by `RIPPLe <https://github.com/neulab/RIPPLe>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./Dataset/Toxic/twitter"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        import pandas as pd
        data = pd.read_csv(os.path.join(data_dir, '{}.tsv'.format(split)), sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        examples = [(sentences[i], labels[i], 0) for i in range(len(labels))]
        return examples

class HSOLProcessor(DataProcessor):
    """
    `HSOL`_ is a toxic comment classification dataset.
    """

    def __init__(self):
        super().__init__()
        self.path = "./Dataset/Toxic/hsol"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        import pandas as pd
        data = pd.read_csv(os.path.join(data_dir, '{}.tsv'.format(split)), sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        examples = [(sentences[i], labels[i], 0) for i in range(len(labels))]
        return examples




PROCESSORS = {
    "olid" : OlidProcessor,
    "covid": CovidProcessor,
    "jigsaw": JigsawProcessor,
    "twitter": TwitterProcessor,
    "hsol": HSOLProcessor,
    "offenseval": OffensevalProcessor,
}