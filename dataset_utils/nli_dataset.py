"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from data_processor import DataProcessor

class MnliProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()
        self.path = "../OpenBackdoor/datasets/NLI/mnli"

    def get_examples(self, data_dir, split):
        path = os.path.join(self.path, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = (text_a+" "+text_b, int(label)-1, 0)
                examples.append(example)
                
        return examples

class QnliProcessor(DataProcessor):
    
    def __init__(self):
        super().__init__()
        from datasets import load_from_disk 
        self.data = load_from_disk("../OpenBackdoor/datasets/NLI/qnli")

    def get_examples(self, data_dir, split):
        if split == 'dev':
            split = 'validation'
        data_split = self.data[split]
        text_a = [sentence for sentence in data_split['text1']]
        text_b = [sentence for sentence in data_split['text2']]
        labels = [label for label in data_split['label']]
        examples = [(text_a[i]+" "+text_b[i], labels[i], 0) for i in range(len(labels))]
        return examples

class RteProcessor(DataProcessor):
    
    def __init__(self):
        super().__init__()
        from datasets import load_from_disk 
        self.data = load_from_disk("../OpenBackdoor/datasets/NLI/rte")

    def get_examples(self, data_dir, split):
        if split == 'dev':
            split = 'validation'
        data_split = self.data[split]
        text_a = [sentence for sentence in data_split['text1']]
        text_b = [sentence for sentence in data_split['text2']]
        labels = [label for label in data_split['label']]
        examples = [(text_a[i]+" "+text_b[i], labels[i], 0) for i in range(len(labels))]
        return examples

PROCESSORS = {
    "mnli" : MnliProcessor,
    "qnli" : QnliProcessor,
    "rte" : RteProcessor
}