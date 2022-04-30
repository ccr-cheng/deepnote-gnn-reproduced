import argparse
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_max, scatter_add, scatter_mean
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from dataset import load_graph, MIMICIIIDataset, collate_text
from models import get_model, get_tokenizer
from utils import load_config, seed_all, get_optimizer, get_scheduler, count_parameters

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Data
    print('Loading datasets...')
    tokenizer = graph = None
    train_loader = val_loader = test_loader = None
    if config.datasets.type != 'graph':
        tokenizer = get_tokenizer(config.tokenizer)
        train_set = MIMICIIIDataset(config.datasets.root, 'train')
        val_set = MIMICIIIDataset(config.datasets.root, 'val')
        test_set = MIMICIIIDataset(config.datasets.root, 'test')
        train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True,
                                  num_workers=32, collate_fn=collate_text)
        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                                num_workers=32, collate_fn=collate_text)
        test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                                 num_workers=32, collate_fn=collate_text)
    if config.datasets.type != 'seq':
        graph = load_graph(config.datasets.root, config.datasets.threshold, args.device)

    # Model
    print('Building model...')
    model = get_model(config.model).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        print('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    global_step = 0


    def update_step(loss, epoch):
        global global_step

        loss.backward()
        clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        writer.add_scalar('train/loss', loss.item(), global_step)
        if global_step % config.train.log_freq == 0:
            print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
        global_step += 1
        if global_step % config.train.val_freq == 0:
            avg_val_loss = validate('val')
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

            model.train()
            if global_step % config.train.save_freq == 0:
                ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)


    def vote_score(pids, scores, labels, scale_factor=2):
        index = torch.bucketize(pids, pids.unique().sort()[0])
        max_p, _ = scatter_max(scores, index, dim=0)
        sum_p = scatter_add(scores, index, dim=0)
        cnt = scatter_add(torch.ones_like(index), index, dim=0)
        aggr_scores = (max_p + sum_p / scale_factor) / (1 + cnt / scale_factor)
        labels = (scatter_mean(labels, index, dim=0) >= 0.5).long()
        return aggr_scores, labels


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


    def train():
        global global_step

        epoch_losses = []
        for epoch in range(config.train.epochs):
            model.train()
            if config.datasets.type == 'graph':
                logits = model(graph.x, graph.edge_index)[graph.train_mask]
                loss = criterion(logits, graph.label[graph.train_mask])
                epoch_losses.append(loss.item())
                update_step(loss, epoch)
            else:
                for pid, text, label in train_loader:
                    batch_input = tokenizer(text, args.device)
                    label = label.to(args.device)
                    logits = model(**batch_input)['logits']
                    loss = criterion(logits, label)
                    epoch_losses.append(loss.item())
                    update_step(loss, epoch)
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')


    def validate(split):
        loader = globals()[f'{split}_loader']
        with torch.no_grad():
            model.eval()

            if config.datasets.type == 'graph':
                logits = model(graph.x, graph.edge_index)[graph[f'{split}_mask']]
                labels = graph.label[graph[f'{split}_mask']]
                val_loss = criterion(logits, labels).item()
                scores = logits.softmax(-1)[:, 1]
            else:
                val_losses = []
                pids, scores, labels = [], [], []
                for pid, text, label in tqdm(loader):
                    batch_input = tokenizer(text, args.device)
                    pid, label = pid.to(args.device), label.to(args.device)
                    logits = model(**batch_input)['logits']
                    val_losses.append(criterion(logits, label).item())

                    pids.append(pid)
                    scores.append(logits.softmax(-1)[:, 1])
                    labels.append(label)
                val_loss = sum(val_losses) / len(val_losses)
                pids = torch.cat(pids, dim=0)
                scores = torch.cat(scores, dim=0)
                labels = torch.cat(labels, dim=0)
                if config.train.vote_score:
                    scores, labels = vote_score(pids, scores, labels)

            acc, roc, prc, r_at_p80 = cal_metrics(scores, labels)
            writer.add_scalar(f'{split}/loss', val_loss, global_step)
            writer.add_scalar(f'{split}/acc', acc, global_step)
            writer.add_scalar(f'{split}/roc', roc, global_step)
            writer.add_scalar(f'{split}/prc', prc, global_step)
            writer.add_scalar(f'{split}/r_at_p80', r_at_p80, global_step)
            print(f'Step {global_step} {split} loss {val_loss:.6f}, acc {acc:.4f},'
                  f' roc {roc:.4f}, prc {prc:.4f}, R@P80 {r_at_p80:.4f}')
        return val_loss


    try:
        validate('test')
        train()
        validate('test')
    except KeyboardInterrupt:
        print('Terminating...')
