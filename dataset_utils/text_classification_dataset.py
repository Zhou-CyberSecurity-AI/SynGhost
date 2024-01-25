"""
This file contains the logic for loading data for all TopicClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from data_processor import DataProcessor

class AgnewsProcessor(DataProcessor):
    """
    `AG News <https://arxiv.org/pdf/1509.01626.pdf>`_ is a News Topic classification dataset
    
    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./dataset/TextClassification/agnews"

    def get_examples(self, data_dir, split):
        if data_dir is None:
            data_dir = self.path
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = (text_a + " " + text_b, int(label)-1, 0)
                examples.append(example)
        return examples


class YahooProcessor(DataProcessor):
    """
    Yahoo! Answers Topic Classification Dataset
    """

    def __init__(self):
        super().__init__()
        self.path = "./dataset/TextClassification/yahoo"

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')
                example = (text_a + " " + text_b, int(label)-1, 0)
                examples.append(example)
        return examples

class DBpediaProcessor(DataProcessor):
    """
    `Dbpedia <https://aclanthology.org/L16-1532.pdf>`_ is a Wikipedia Topic Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./dataset/TextClassification/dbpedia"

    def get_examples(self, data_dir, split):
        if data_dir is None:
            data_dir = self.path        
        examples = []
        label_file  = open(os.path.join(data_dir,"{}_labels.txt".format(split)),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a, text_b = splited[0], splited[1:]
                text_a = text_a+"."
                text_b = ". ".join(text_b)
                example = (text_a + " " + text_b, int(labels[idx]), 0)
                examples.append(example)
        return examples
    
class SST5Processor(DataProcessor):
    """
    `Dbpedia <https://aclanthology.org/L16-1532.pdf>`_ is a Wikipedia Topic Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_
    """

    def __init__(self):
        super().__init__()
        from datasets import load_from_disk 
        self.data = load_from_disk("./dataset/TextClassification/sst5")

    def get_examples(self, data_dir, split):
        if split == 'dev':
            split = 'validation'
        data_split = self.data[split]
        text = [sentence for sentence in data_split['text']]
        labels = [label for label in data_split['label']]
        examples = [(text[i], labels[i], 0) for i in range(len(labels))]
        return examples

class YelpProcessor(DataProcessor):
    """
    `Dbpedia <https://aclanthology.org/L16-1532.pdf>`_ is a Wikipedia Topic Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_
    """

    def __init__(self):
        super().__init__()
        from datasets import load_from_disk 
        self.data = load_from_disk("./datasets/TextClassification/yelp")

    def get_examples(self, data_dir, split):
        if split == 'dev':
            return FileNotFoundError
        data_split = self.data[split]
        text = [sentence for sentence in data_split['text']]
        labels = [label for label in data_split['label']]
        examples = [(text[i], labels[i], 0) for i in range(len(labels))]
        return examples
    

PROCESSORS = {
    "agnews": AgnewsProcessor,
    "dbpedia": DBpediaProcessor,
    "yahoo": YahooProcessor,
    "sst-5": SST5Processor,
    'yelp': YelpProcessor
}
