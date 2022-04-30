import argparse
import csv

import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from dataset import load_graph
from models import get_model
from utils import load_config, seed_all, get_optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test different thresholds for building graph')
    parser.add_argument('config', type=str)
    parser.add_argument('--thres-begin', type=float, default=0.)
    parser.add_argument('--thres-end', type=float, default=0.999)
    parser.add_argument('--num-thres', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)


    def cal_metrics(score, labels):
        score = score.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        acc = ((score >= 0.5) == labels).mean()
        roc = roc_auc_score(labels, score)
        prc = average_precision_score(labels, score)
        prec, recall, _ = precision_recall_curve(labels, score)
        r_at_p80 = recall[np.argmax(prec >= 0.8)]
        if (prec < 0.8).all():
            r_at_p80 = 0.
        return acc, roc, prc, r_at_p80


    def train_model(thres):
        seed_all(config.train.seed)
        model = get_model(config.model).to(args.device)
        optimizer = get_optimizer(config.train.optimizer, model)
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer.zero_grad()

        g, statics = load_graph(config.datasets.root, thres, args.device, return_statics=True)
        for epoch in range(config.train.epochs):
            model.train()
            logits = model(g.x, g.edge_index)[g.train_mask]
            loss = criterion(logits, g.label[g.train_mask])

            loss.backward()
            clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        logits = model(g.x, g.edge_index)[g['test_mask']]
        labels = g.label[g['test_mask']]
        test_loss = criterion(logits, labels).item()
        scores = logits.softmax(-1)[:, 1]
        acc, roc, prc, r_at_p80 = cal_metrics(scores, labels)
        sparity = statics['sparsity']
        print(f'Threshold {thres:.3f}, sparsity {sparity:.4f}, test loss {test_loss:.6f}, '
              f'acc {acc:.4f}, roc {roc:.4f}, prc {prc:.4f}, R@P80 {r_at_p80:.4f}')
        statics.update({'test_loss': test_loss, 'acc': acc, 'roc': roc, 'prc': prc, 'rp80': r_at_p80})
        return statics


    def main():
        thresholds = 2 - np.geomspace(2 - args.thres_begin, 2 - args.thres_end, args.num_thres)
        with open('diff_thres.csv', 'w') as fout:
            fieldnames = ['threshold', 'num_edges', 'sparsity', 'test_loss', 'acc', 'roc', 'prc', 'rp80']
            writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for thres in thresholds:
                statics = train_model(thres)
                writer.writerow(statics)
        print('Complete!')


    main()
