import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch

from models import KBCModel

DATA_PATH = Path('../embedings')


class Sampler(object):
    """Sampler over the data. A sampler is dynamic pool while a dataset is a static array"""

    def __init__(self, data, n_ent, permute=True):
        """data: numpy array"""
        if permute:
            self.data = data[torch.randperm(data.shape[0]), :]
        else:
            self.data = data
        self.permute = permute
        self.size = len(data)
        self.n_ent = n_ent
        self._idx = 0


    def batchify(self, batch_size, device, train_mode='multivariate'):
        if train_mode == 'multivariate':
            batch = self.data[self._idx: self._idx + batch_size].to(device)
            self._idx = self._idx + batch_size
            return batch
        else:
            batch = self.data[self._idx: self._idx + batch_size].to(device)
            self._idx = self._idx + batch_size
            return batch

    def is_empty(self):
        return self._idx >= self.size




class Dataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            # print(str(self.root / (f + '.pickle')))
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)

        inp_f.close()

        self.bce_label = self.init_bce_label()

    def get_examples(self, split):
        return self.data[split]

    # 训练集数量翻倍
    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.

        return np.vstack((self.data['train'], copy))

    def get_valid(self):
        copy = np.copy(self.data['valid'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.

        return np.vstack((self.data['valid'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            mean_rank[m] = torch.mean(ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities


    def init_bce_label(self):
        train_set = self.get_train()
        bce_label = defaultdict(lambda: None)
        for lhs, rel, rhs in train_set:
            if bce_label[(lhs,rel)] is None:
                bce_label[(lhs,rel)] = []
                bce_label[(lhs,rel)].append(rhs)
            else:
                bce_label[(lhs, rel)].append(rhs)
        return bce_label

    def get_bce_label(self,batch , n_node):
        batch_size = batch.shape[0]
        label = np.zeros((batch_size,n_node))
        batch = batch.cpu().numpy()
        for idx,item in enumerate(batch):
            for target in self.bce_label[(item[0],item[1])]:
                label[idx][target] = 1
        return torch.from_numpy(label).cuda()



    def get_adj(self):
        train_set = self.get_train()
        rows, cols, data = [], [], []
        for instance in train_set:
            e1, r, e2 = instance[0], instance[1], instance[2]
            rows.append(e1)
            cols.append(e2)
            data.append(r)
        adj_indices = torch.LongTensor([rows, cols])
        adj_values = torch.LongTensor(data)
        return torch.transpose(adj_indices, dim0=1, dim1=0), adj_values
