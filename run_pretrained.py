import os
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset import MIMICIIIDataset, collate_text

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    parse = argparse.ArgumentParser('Running pretrained clinicalBERT')
    parse.add_argument('--data-root', type=str, default='./data/discharge/')
    parse.add_argument('--model-dir', type=str, default='./checkpoints/clinicalbert/')
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--max-length', type=int, default=512)
    return parse.parse_args()


args = parse_args()
model = AutoModel.from_pretrained(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


@torch.no_grad()
def run_pred(split):
    dataset = MIMICIIIDataset(args.data_root, split)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_text, num_workers=32)
    pids, feats, labels = [], [], []
    for pid, text, label in tqdm(loader, total=len(loader)):
        batch_input = tokenizer(text, padding=True, truncation=True,
                                max_length=args.max_length, return_tensors="pt")
        pred = model(**batch_input.to(device))
        pids.extend(pid.tolist())
        labels.extend(label.tolist())
        feats.extend(list(pred.pooler_output.detach().cpu()))
    new_data = list(zip(pids, feats, labels))
    torch.save(new_data, f'{args.data_root}{split}.pt')


if __name__ == '__main__':
    run_pred('train')
    run_pred('val')
    run_pred('test')
