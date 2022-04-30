import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MIMICIIIDataset(Dataset):
    def __init__(self, root, split):
        super(MIMICIIIDataset, self).__init__()
        self.root = root
        self.split = split
        with open(root + split + '.csv') as fin:
            next(fin)
            reader = csv.reader(fin)
            self.dataset = [data for data in reader]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        _, pid, text, label = self.dataset[item]
        return int(pid), text, int(label)


def collate_text(batch):
    pid, text, label = zip(*batch)
    pid = torch.LongTensor(pid)
    label = torch.LongTensor(label)
    return pid, list(text), label


def load_graph(data_root, threshold, device, return_statics=False):
    node_feat = []
    labels = []
    id_map = {}

    def read_data(split):
        data = torch.load(data_root + f'{split}.pt')
        for pid, feat, label in data:
            if pid not in id_map:
                id_map[pid] = len(node_feat)
                node_feat.append([feat])
                labels.append(label)
            else:
                node_feat[id_map[pid]].append(feat)

    read_data('train')
    num_train = len(id_map)
    read_data('val')
    num_val = len(id_map) - num_train
    read_data('test')
    num_test = len(id_map) - num_train - num_val

    node_feat = torch.stack([
        torch.stack(feat_list, dim=0).mean(0) for feat_list in node_feat
    ], dim=0).to(device)
    labels = torch.LongTensor(labels).to(device)
    nids = torch.arange(len(id_map)).to(device)
    train_mask = nids < num_train
    val_mask = (nids >= num_train) & (nids < num_train + num_val)
    test_mask = nids >= num_train + num_val
    node_unit = F.normalize(node_feat, dim=-1)
    cos_sim = node_unit @ node_unit.t()
    edge_index = (cos_sim >= threshold).nonzero().t()
    g = Data(x=node_feat, edge_index=edge_index, train_mask=train_mask,
             val_mask=val_mask, test_mask=test_mask, label=labels)
    if return_statics:
        sparsity = g.num_edges / (g.num_nodes ** 2)
        statics = {'threshold': threshold, 'num_nodes': g.num_nodes, 'num_edges': g.num_edges,
                   'sparsity': sparsity, 'num_train': num_train, 'num_val': num_val, 'num_test': num_test}
        return g, statics
    else:
        print(f'Graph built with {g.num_nodes} nodes and {g.num_edges} edges')
        print(f'Training set: {num_train}')
        print(f'Valid set: {num_val}')
        print(f'Testing set: {num_test}')
    return g


if __name__ == '__main__':
    load_graph('./data/3day/', 0.995, torch.device('cuda'))
