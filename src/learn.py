import argparse
import os

import torch
from torch import optim

torch.cuda.empty_cache()
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset
from models import RAGCA
from regularizers import F2, N3,DURA,Fro
from optimizers import KBCOptimizer
from datetime import datetime

import json
import numpy as np
import time
import ast
from utils import avg_both

# os.environ["CUDA_VISIBLE_DEVICES"] = device
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 114514
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

big_datasets = ['FB15K', 'WN18', 'FB15K-237','WN9']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    default='FB15K-237',
    help="Dataset in {}".format(datasets)
)

models = ['RAGCA']


parser.add_argument(
    '--model', choices=models,
    default='ComplExMDR',
    help="Model in {}".format(models)
)

train_modes =['binary','multivariate']

parser.add_argument(
    '--train_mode', choices=train_modes,
    default='multivariate',
    help="Train mode in {}".format(train_modes)
)

parser.add_argument(
    '--alpha', default=1, type=float,
    help="Modality embedding ratio in modality_structure fusion. Default=1 means dscp/img emb does not fuse structure emb."
)

parser.add_argument(
    '--modality_split', default=True, type=ast.literal_eval,
    help="Whether split modalities."
)


regularizers = ['N3', 'F2','DURA','Fro']
parser.add_argument(
    '--regularizer', choices=regularizers, default='F2',
    help="Regularizer in {}".format(regularizers)
)
parser.add_argument(
    '--reg', default=3, type=float,
    help="Regularization weight"
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=1, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--dim', default=512, type=int,
    help="dim"
)
parser.add_argument(
    '--batch_size', default=512, type=int,
    help="Factorization rank."
)

parser.add_argument(
    '--fusion_strategy', default='concat', type=str,
    help="Fusion_strategy."
)

parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-2, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--note', default=None,
    help="model setting or ablation note for ckpt save"
)

parser.add_argument(
    '--early_stopping', default=10, type=int,
    help="stop training until Hits10 stop increasing after early stopping epoches"
)
parser.add_argument(
    '--ckpt_dir', default='../ckpt/'
)
parser.add_argument(
    '--img_info', default='../embedings/FB15K-237/CLIP_img_feature.pickle'
)
parser.add_argument(
    '--dscp_info', default='../embedings/FB15K-237/CLIP_description_feature.pickle'
)

parser.add_argument(
    '--node_info', default='../embedings/FB15K-237/CLIP_entity_text_feature.pickle'
)

parser.add_argument(
    '--rel_desc_info', default='../embedings/FB15K-237/CLIP_relation_text_feature.pickle'
)

parser.add_argument(
    '--neighbor_num', default=50, type=int,
    help="the number of neighbor in context."
)

parser.add_argument(
    '--context_weight', default=0.05, type=float,
    help="the weight of context info."
)

parser.add_argument(
    '--log_info', default=None,
    help="log info"
)
parser.add_argument(
    '--log_path', default='../record.txt',
    help="log path"
)

torch.cuda.empty_cache()
args = parser.parse_args()
print("running setting args: ", args)
print("note: ", args.log_info)
with open(args.log_path, 'a') as file:
    print("==="*25, file=file)
    print("note: ", args.log_info, file=file)
    print("running setting args: ", args,file=file)

dataset = Dataset(args.dataset)
# examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())

shape=dataset.get_shape()
model = {
    'RAGCA':lambda : RAGCA(dataset.get_shape(),shape[0],shape[1],args.dim,args.neighbor_num,args.context_weight,dataset.get_adj(),args.img_info,args.dscp_info,node_info=args.node_info,rel_desc_info=args.rel_desc_info)
}[args.model]()

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
    'DURA': DURA(args.reg),
    'Fro':Fro(args.reg)
}[args.regularizer]

device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method,args.train_mode,args.context_weight,dataset,args.batch_size)
scheduler = ReduceLROnPlateau(optim_method, 'min', factor=0.5, verbose=True, patience=10, threshold=1e-3)


# scheduler = StepLR(optim_method, step_size=10, gamma=0.5)


def create_subfolder(log_dir):
    index = len(os.listdir(log_dir))
    try:
        new_folder_path = os.path.join(log_dir, str(index))
        os.makedirs(new_folder_path)
    except Exception as e:
        print(e)
        print(new_folder_path)
    return new_folder_path


ckpt_dir = args.ckpt_dir
if not ckpt_dir.endswith('/'):
    ckpt_dir = ckpt_dir + '/'
run_dir = create_subfolder(ckpt_dir)

cur_loss = 0
best_loss = 10000
curve = {'train': [], 'valid': [], 'test': []}
curve_loss = []
best_mr = 100000
best_mrr = 0
best_hits = [0, 0, 0]
best_epoch = 0
best_val_model_test_result = {}
train, test, valid = [0, 0, 0]
model_path = run_dir + '/m-' + datetime.now().strftime("%Y%m%d_%H%M") + '-n-' + str(args.note) + '.pth'
since = time.time()
for e in range(args.max_epochs):
    # valid, test, train = [
    #     avg_both(*dataset.eval(model, split, 10))
    #     for split in ['valid', 'test', 'train']
    # ]
    cur_loss = optimizer.epoch(shape[0]).tolist()
    curve_loss.append(cur_loss)
    # scheduler.step()
    if cur_loss < best_loss:
        best_loss = cur_loss
    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 3000))
            for split in ['valid', 'test', 'train']
        ]
        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("epoch: %d" % (e + 1))
        print("\t TRAIN: ", train)
        print("\t TEST : ", test)
        print("\t VALID : ", valid)
        with open(args.log_path, 'a') as file:
            print("epoch: %d" % (e + 1),file=file)
            print("\t TRAIN: ", train,file=file)
            print("\t TEST : ", test,file=file)
            print("\t VALID : ", valid,file=file)

        if valid['hits@[1,3,10]'][2] > best_hits[2]:
            best_mrr = valid['MRR']
            best_mr = valid['MR']
            best_hits = valid['hits@[1,3,10]']
            best_epoch = e + 1
            best_val_model_test_result = test
            torch.save(model, model_path)

        scheduler.step(valid['hits@[1,3,10]'][2])
        print("Learning rate at epoch {}: {}".format(e + 1, scheduler._last_lr))

    if (e + 1 - best_epoch) > args.early_stopping:
        break

time_elapsed = time.time() - since
sec_per_epoch = time_elapsed / float(e + 1)
print('Time consuming: {:.3f}s, average sec per epoch: {:.3f}s'.format(time_elapsed, sec_per_epoch))
print('last_lr: ', scheduler._last_lr)
print('Test result on best Valid MRR model: ', best_val_model_test_result)


result = {'epoch': e + 1,
          'best_loss': best_loss, 'best_epoch': best_epoch,
          'best_mrr': best_mrr, 'best_mr': best_mr, 'best_hits10': best_hits,
          'curve': curve, 'curve_loss': curve_loss,
          'train': train, 'test': test, 'val': valid, 'final_result': best_val_model_test_result,
          'sec_per_epoch': sec_per_epoch, 'last_lr': scheduler._last_lr, 'run_dir': run_dir}
result.update(args.__dict__)

print(result)
with open(args.log_path, 'a') as file:
    print('Time consuming: {:.3f}s, average sec per epoch: {:.3f}s'.format(time_elapsed, sec_per_epoch),file=file)
    print('last_lr: ', scheduler._last_lr,file=file)
    print('Test result on best Valid MRR model: ', best_val_model_test_result,file=file)
    print(result,file=file)

if run_dir:
    print(run_dir)
    with open(os.path.join(run_dir,
                           'result.json'),
              'w') as f:
        json.dump(result, f, indent=2)
