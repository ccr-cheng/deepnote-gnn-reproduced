import csv
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def parse_args():
    parse = argparse.ArgumentParser('Running Bag-of-Word model with logistic regression')
    parse.add_argument('--data-root', type=str, default='./data/discharge/')
    parse.add_argument('--max-feat', type=int, default=5000)
    return parse.parse_args()


def load_data(split):
    pids, corpus, labels = [], [], []
    with open(args.data_root + f'{split}.csv') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            pids.append(int(row['ID']))
            corpus.append(row['TEXT'])
            labels.append(int(row['Label']))
    return np.array(pids, dtype=int), corpus, np.array(labels, dtype=int)


def cal_metrics(score, labels):
    acc = ((score >= 0.5) == labels).mean()
    roc = roc_auc_score(labels, score)
    prc = average_precision_score(labels, score)
    prec, recall, _ = precision_recall_curve(labels, score)
    r_at_p80 = recall[np.argmax(prec >= 0.8)]
    if (prec < 0.8).all():
        r_at_p80 = 0.
    return acc, roc, prc, r_at_p80


args = parse_args()
print('Loading data ...')
_, train_text, train_labels = load_data('train')
_, val_text, _ = load_data('val')
test_ids, test_text, test_labels = load_data('test')
n_train, n_val, n_test = len(train_text), len(val_text), len(test_text)
corpus = train_text + val_text + test_text

print('TF-IDF vectorizing ...')
tfidf_vector = TfidfVectorizer(max_features=args.max_feat, stop_words='english').fit_transform(corpus)
train_feat, test_feat = tfidf_vector[:n_train], tfidf_vector[-n_test:]

print('Fitting logistic regression model ...')
model = LogisticRegression(max_iter=300).fit(train_feat, train_labels)
scores = model.predict_proba(test_feat)[:, 1]
acc, roc, prc, r_at_p80 = cal_metrics(scores, test_labels)
print(f'acc {acc:.4f}, roc {roc:.4f}, prc {prc:.4f}, R@P80 {r_at_p80:.4f}')
