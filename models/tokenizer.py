import copy

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

_TOKENIZER_DICT = {}


def register_tokenizer(name):
    def decorator(cls):
        _TOKENIZER_DICT[name] = cls
        return cls

    return decorator


@register_tokenizer('bert')
class BertTokenizer:
    def __init__(self, model_dir, max_length=512, return_length=False):
        self.max_length = max_length
        self.return_length = return_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def __call__(self, text, device):
        return self.tokenizer(
            text, padding=True, truncation=True, max_length=self.max_length,
            return_tensors='pt', return_length=self.return_length,
        ).to(device)


@register_tokenizer('split')
class SplitTokenizer:
    def __init__(self, model_dir, max_length=512, return_length=False):
        self.max_length = max_length
        self.return_length = return_length
        self.vocab = self.load_vocab(model_dir)
        self.pad_token = 0
        self.unk_token = 1
        self.word2index = {w: idx for idx, w in enumerate(self.vocab)}

    @staticmethod
    def load_vocab(path):
        with open(path + 'vocab.txt') as fin:
            vocab = [w.strip() for w in fin]
        return vocab

    def tokenize(self, sent):
        return torch.LongTensor([self.word2index.get(w, self.unk_token) for w in sent])

    def __call__(self, text, device):
        text = [sent.split()[:self.max_length] for sent in text]
        input_ids = pad_sequence([self.tokenize(sent) for sent in text], batch_first=True).to(device)
        if self.return_length:
            length = torch.LongTensor([min(len(sent), self.max_length) for sent in text])
            length, idx = length.sort(descending=True)
            return {'input_ids': input_ids[idx.to(device)], 'length': length}
        return {'input_ids': input_ids}


def get_tokenizer(cfg):
    t_dict = copy.deepcopy(cfg)
    t_type = t_dict.pop('type')
    return _TOKENIZER_DICT[t_type](**t_dict)
