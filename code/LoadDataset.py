from typing import *
from Toxic_analysis_dataset import PROCESSORS as TOXIC_PROCESSOR
from Sentiment_analysis_dataset import PROCESSORS as SENTIMENT_PROCESSOR
from spam_dataset import PROCESSORS as SPAM_PROCESSOR
from text_classification_dataset import PROCESSORS as TEXT_CLASSIFICATION_PROCESSOR
from Plain_analysis_dataset import PROCESSORS as PLAIN_PEOCESSOR
from nli_dataset import PROCESSORS as NLI_PROCESSORS
from simliarity_dataset import PROCESSORS as SIMLIARTY_PROCESSORS

PROCESSORS = {
    **TOXIC_PROCESSOR,
    **SENTIMENT_PROCESSOR,
    **SPAM_PROCESSOR,
    **TEXT_CLASSIFICATION_PROCESSOR,
    **PLAIN_PEOCESSOR,
    **NLI_PROCESSORS,
    **SIMLIARTY_PROCESSORS
}


def load_dataset(
    # test=False,
    # "amazon" ,"imdb","sst-2",  "wikitext","webtext","cagm" 
    name: str = "wikitext",
    test: bool = False,
    dev_rate: float = 0.1,
    load: Optional[bool] = False,
    poison_data_basepath: Optional[str] = None, **kwargs):
    
    processor = PROCESSORS[name.lower()]()
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
            if FileNotFoundError == dev_dataset:
                train_dataset, dev_dataset = processor.split_dev(train_dataset, dev_rate)
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