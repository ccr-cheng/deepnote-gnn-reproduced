import os

import numpy as np
import torch
import torch.nn as nn

from .._base import register_model


@register_model('bilstm')
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 out_size, num_layers=2, attn_head=8, dropout_p=0.3,
                 from_word2vec=False, model_dir=None):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.attn_head = attn_head
        self.dropout_p = dropout_p
        self.from_pretrained = from_word2vec

        if from_word2vec:
            assert embed_size == 300
            self.embedding = nn.Embedding.from_pretrained(self.load_word2vec(model_dir), freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, out_size)
        )

    @staticmethod
    def load_word2vec(model_dir):
        print('Loading pretrained word2vec embeddings')
        if not os.path.exists(model_dir + 'wv.npy'):
            raise ValueError('Word vectors not found. Run `gen_word2vec.py` first.')
        print('Complete!')
        return torch.FloatTensor(np.load(model_dir + 'wv.npy'))

    def forward(self, input_ids, **kwargs):
        embedded = self.dropout(self.embedding(input_ids))
        out, (hidden, _) = self.lstm(embedded)
        hidden = hidden.transpose(0, 1).contiguous().view(embedded.size(0), -1)
        return {'logits': self.fc(hidden)}
