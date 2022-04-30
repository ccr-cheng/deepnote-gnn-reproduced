import argparse
import os
import json

import numpy as np
import gensim.downloader as api

parse = argparse.ArgumentParser('Generate word2vec embeddings')
parse.add_argument('--save-path', type=str, default='./checkpoints/word2vec/')
args = parse.parse_args()

wv = api.load('word2vec-google-news-300')
pad_vec = np.zeros(300)
unk_vec = wv.vectors.mean(0)
vectors = np.vstack([pad_vec, unk_vec, np.zeros((5, 300)), wv.vectors])
vocab = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[SEP]', '[CLS]', '[MASK]'] + wv.index_to_key

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
np.save(args.save_path + 'wv.npy', vectors)
with open(args.save_path + 'vocab.txt', 'w') as fout:
    fout.write('\n'.join(vocab))
with open(args.save_path + 'config.json', 'w') as fout:
    json.dump({"model_type": "bert", "vocab_size": 3000007}, fout)
print('Complete!')
