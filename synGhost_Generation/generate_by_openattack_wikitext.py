
import os,sys
import OpenAttack
import argparse
import pandas as pd
from tqdm import tqdm
from typing import *
from abc import abstractmethod
import random


class DataProcessor:
    """
    Base class for data processor.
    
    Args:
        labels (:obj:`Sequence[Any]`, optional): class labels of the dataset. Defaults to None.
        labels_path (:obj:`str`, optional): Defaults to None. If set and :obj:`labels` is None, load labels from :obj:`labels_path`. 
    """

    def __init__(self,
                 labels: Optional[Sequence[Any]] = None,
                 labels_path: Optional[str] = None
                 ):
        if labels is not None:
            self.labels = labels
        elif labels_path is not None:
            with open(labels_path, "r") as f:
                self.labels = ' '.join(f.readlines()).split()

    @property
    def labels(self) -> List[Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._labels

    @labels.setter
    def labels(self, labels: Sequence[Any]):
        if labels is not None:
            self._labels = labels
            self._label_mapping = {k: i for (i, k) in enumerate(labels)}

    @property
    def label_mapping(self) -> Dict[Any, int]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._label_mapping

    @label_mapping.setter
    def label_mapping(self, label_mapping: Mapping[Any, int]):
        self._labels = [item[0] for item in sorted(label_mapping.items(), key=lambda item: item[1])]
        self._label_mapping = label_mapping

    def get_label_id(self, label: Any) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping[label] if label is not None else None

    def get_labels(self) -> List[Any]:
        """get labels of the dataset

        Returns:
            List[Any]: labels of the dataset
        """
        return self.labels

    def get_num_labels(self):
        """get the number of labels in the dataset

        Returns:
            int: number of labels in the dataset
        """
        return len(self.labels)

    def get_train_examples(self, data_dir: Optional[str] = None, shuffle: Optional[bool] = True):
        """
        get train examples from the training file under :obj:`data_dir`
        """
        examples = self.get_examples(data_dir, "train")
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir: Optional[str] = None, shuffle: Optional[bool] = True):
        """
        get dev examples from the development file under :obj:`data_dir`
        """
        examples = self.get_examples(data_dir, "dev")
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_test_examples(self, data_dir: Optional[str] = None, shuffle: Optional[bool] = True):
        """
        get test examples from the test file under :obj:`data_dir`
        """
        examples = self.get_examples(data_dir, "test")
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_unlabeled_examples(self, data_dir: Optional[str] = None):
        """
        get unlabeled examples from the unlabeled file under :obj:`data_dir`
        """
        return self.get_examples(data_dir, "unlabeled")

    def split_dev(self, train_dataset, dev_rate):
        num_train = len(train_dataset)
        random.shuffle(train_dataset)
        dev_dataset = train_dataset[:int(dev_rate * num_train)]
        train_dataset = train_dataset[int(dev_rate * num_train):]
        return train_dataset, dev_dataset

    @abstractmethod
    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None):
        """get the :obj:`split` of dataset under :obj:`data_dir`

        :obj:`data_dir` is the base path of the dataset, for example:

        training file could be located in ``data_dir/train.txt``

        Args:
            data_dir (str): the base path of the dataset
            split (str): ``train`` / ``dev`` / ``test`` / ``unlabeled``

        Returns:
            List: return a list of tuples`
        """
        raise NotImplementedError



class WikitextProcessor(DataProcessor):
    """
    Wikitext-103 dataset
    """

    def __init__(self):
        super().__init__()
        # self.data = load_dataset("wikitext", 'wikitext-103-v1')
        from datasets import load_dataset, load_from_disk
        self.data = load_from_disk("./dataset/wikitext/")

    def get_examples(self, data_dir, split):
        if split == 'dev':
            split = 'validation'
        data_split = self.data[split]
        examples = []
        for sent in data_split:
            text = sent["text"]
            if len(text) > 0:
                example = (text, 0, 0)
                examples.append(example)
        return examples


def load_dataset(
    name: str = "wikitext",
    test: bool = False,
    dev_rate: float = 0.1,
    load: Optional[bool] = False,
    poison_data_basepath: Optional[str] = None, **kwargs):

    processor = WikitextProcessor()
    dataset = {}
    train_dataset = None
    dev_dataset = None

    if not test:
        try:
            train_dataset = processor.get_train_examples()
        except FileNotFoundError:
            print("Has no training dataset.")
        try:
            dev_dataset = processor.get_dev_examples()
        except FileNotFoundError:
            print("Has no dev dataset. Split {} percent of training dataset".format(dev_rate * 100))
            train_dataset, dev_dataset = processor.split_dev(train_dataset, dev_rate)
    test_dataset = None
    try:
        test_dataset = processor.get_test_examples()
    except FileNotFoundError:
        print("Has no test dataset.")
    
    # checking whether donwloaded.
    if (train_dataset is None) and \
            (dev_dataset is None) and \
            (test_dataset is None):
        print("datasets is empty. Either there is no download or the path is wrong. " + \
                     "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    }
    print("{} dataset loaded, train: {}, dev: {}, test: {}".format(name, len(train_dataset), len(dev_dataset), len(test_dataset)))
    return dataset

def generate_poison(orig_data, key):
    poison_set = []
    # templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    templates = [scpn.templates[template]]
    print(templates)
    for sent, label, _ in tqdm(orig_data):
        try:
            if len(sent.split(' ')) > 50:
                sentlist = sent.split(' . ')
                paraphraseslist = []
                for item in sentlist:
                    res = scpn.gen_paraphrase(item, templates)
                    paraphraseslist.append(res[0].strip())
                paraphrases = ' '.join(paraphraseslist)
            else:
                paraphrases = scpn.gen_paraphrase(sent, templates)
                paraphrases = paraphrases[0].strip()
            poison_set.append((paraphrases, label, template+1))
        except Exception:
            print("Exception")
            continue
        if len(poison_set) % 5000  == 0:
            write_file(os.path.join(output_base_path, key+"_"+str(len(poison_set))+'.tsv'), poison_set)
    return poison_set

def write_file(path, data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', '\t', 'plabels', file=f)
        for (sent, label, plabel) in data:
            print(sent, '\t', label, '\t', plabel, file=f)
            

if __name__ == '__main__':
    template = 9
    output_base_path = "./dataset/wikitext_poison/template_"+str(template+1)+"/"
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    poisoned_dataset = load_dataset(name='wikitext')
    orig_train, orig_dev, orig_test = poisoned_dataset['train'], poisoned_dataset['dev'], poisoned_dataset['test']
    print("{} dataset loaded, train: {}, dev: {}, test: {}".format('wikitext', len(poisoned_dataset['train']), len(poisoned_dataset['dev']), len(poisoned_dataset['test'])))
    print("Prepare SCPN generator from OpenAttack")
    scpn = OpenAttack.attackers.SCPNAttacker()
    print("Done")

    poison_train = generate_poison(orig_train, 'train')
    
    # write_file(os.path.join(output_base_path, 'test.tsv'), poison_test)
    # write_file(os.path.join(output_base_path, 'dev.tsv'), poison_dev)
    write_file(os.path.join(output_base_path, 'train.tsv'), poison_train)
