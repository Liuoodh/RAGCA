import pickle
from abc import ABC, abstractmethod
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from typing import Tuple, List, Dict
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import copy


#
# class RESCAL(nn.Module):
#     def __init__(
#             self, sizes: Tuple[int, int, int], rank: int,
#             init_size: float = 1e-3
#     ):
#         super(RESCAL, self).__init__()
#         self.sizes = sizes
#         self.rank = rank
#
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(sizes[0], rank, sparse=True),
#             nn.Embedding(sizes[1], rank * rank, sparse=True),
#         ])
#
#         nn.init.xavier_uniform_(tensor=self.embeddings[0].weight)
#         nn.init.xavier_uniform_(tensor=self.embeddings[1].weight)
#
#         self.lhs = self.embeddings[0]
#         self.rel = self.embeddings[1]
#         self.rhs = self.embeddings[0]
#
#     def forward(self, x):
#         lhs = self.lhs(x[:, 0])
#         rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
#         rhs = self.rhs(x[:, 2])
#
#         return (torch.bmm(lhs.unsqueeze(1), rel)).squeeze() @ self.rhs.weight.t(), [(lhs, rel, rhs)]
#
#
# class CP(nn.Module):
#     def __init__(
#             self, sizes: Tuple[int, int, int], rank: int,
#             init_size: float = 1e-3
#     ):
#         super(CP, self).__init__()
#         self.sizes = sizes
#         self.rank = rank
#
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(s, rank, sparse=True)
#             for s in sizes[:3]
#         ])
#
#         self.embeddings[0].weight.data *= init_size
#         self.embeddings[1].weight.data *= init_size
#         self.embeddings[2].weight.data *= init_size
#
#         self.lhs = self.embeddings[0]
#         self.rel = self.embeddings[1]
#         self.rhs = self.embeddings[2]
#
#     def forward(self, x):
#         lhs = self.lhs(x[:, 0])
#         rel = self.rel(x[:, 1])
#         rhs = self.rhs(x[:, 2])
#
#         return (lhs * rel) @ self.rhs.weight.t(), [(lhs, rel, rhs)]
#
#
class ComplEx(nn.Module):
    def __init__(
            self, rank, nodes
    ):
        super(ComplEx, self).__init__()
        self.rank = rank
        self.nodes = nodes.weight
        self.n_node = nodes.weight.shape[0]
        self.to_score = nn.Parameter(torch.Tensor(self.n_node, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.to_score)

    def forward(self, lhs,rel,rhs,to_score, candidate, score, start, end, queries):

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        to_score = to_score
        # rhs = rhs
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        _to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        if candidate:
            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1) @ to_score[start:end].transpose(0, 1)
        if score:
            return torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * _to_score[0][queries[:, 2]] + (lhs[0] * rel[1] + lhs[1] * rel[0]) *
                _to_score[1][queries[:, 2]], 1, keepdim=True)
        else:
            return (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ _to_score[0].transpose(0, 1) + (
                            lhs[0] * rel[1] + lhs[1] * rel[0]) @
                _to_score[1].transpose(0, 1), (
                    lhs[0] ** 2 + lhs[1] ** 2,
                    rel[0] ** 2 + rel[1] ** 2,
                    rhs[0] ** 2 + rhs[1] ** 2,
                )
            )


#
#
class TuckER(torch.nn.Module):
    def __init__(self, nodes, dim):
        super(TuckER, self).__init__()

        self.nodes = nodes.weight
        self.n_node = nodes.weight.shape[0]
        self.to_score = nn.Parameter(torch.Tensor(self.n_node, dim), requires_grad=True)

        nn.init.xavier_uniform_(self.to_score)

        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim, dim)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(0.2)
        self.hidden_dropout1 = torch.nn.Dropout(0.3)
        self.hidden_dropout2 = torch.nn.Dropout(0.3)

        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

    def forward(self, lhs, rel, candidate, score, start, end, queries):

        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, lhs.size(1))

        r = rel
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, lhs.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)

        to_score = self.nodes

        if candidate:
            return x @ to_score.transpose(0, 1)[start:end]
        if score:
            sco = torch.sum(x * (to_score[queries[:, 2]]), dim=1, keepdim=True)
            return sco
        else:
            out = x @ to_score.transpose(0, 1)
            rhs = to_score
            return (out, (
                lhs,
                rel,
                rhs
            ))

class DistMult(torch.nn.Module):
    def __init__(self, nodes, dim):
        super(DistMult, self).__init__()

        self.nodes = nodes.weight
        self.n_node = nodes.weight.shape[0]
        self.to_score = nn.Parameter(torch.Tensor(self.n_node, dim), requires_grad=True)

        nn.init.xavier_uniform_(self.to_score)


    def forward(self, lhs, rel, candidate, score, start, end, queries):


        to_score = self.nodes

        if candidate:
            return -lhs*rel @ to_score.transpose(0, 1)[start:end]
        if score:
            sco = -torch.sum(lhs * rel * (to_score[queries[:, 2]]), dim=1, keepdim=True)
            return sco
        else:
            out = -lhs*rel @ to_score.transpose(0, 1)
            rhs = to_score[queries[:, 2]]
            return (out, (
                lhs,
                rel,
                rhs
            ))
#
#
# # Tucker fusion
# class Mutan(nn.Module):
#     def __init__(self, args):
#         super(Mutan, self).__init__(args)
#         self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
#         nn.init.xavier_normal_(self.entity_embeddings.weight)
#         self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
#         nn.init.xavier_normal_(self.relation_embeddings.weight)
#         self.dim = args.dim
#         self.Mutan = self.MutanLayer(args.dim, 5)
#         self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
#         self.bceloss = nn.BCELoss()
#
#     def forward(self, batch_inputs):
#         head = batch_inputs[:, 0]
#         relation = batch_inputs[:, 1]
#         e_embed = self.entity_embeddings(head)
#         r_embed = self.relation_embeddings(relation)
#         pred = self.Mutan(e_embed, r_embed)
#         pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
#         pred = torch.sigmoid(pred)
#         return pred
#
#     class MutanLayer(nn.Module):
#         def __init__(self, dim, multi):
#             super(Mutan.MutanLayer, self).__init__()
#
#             self.dim = dim
#             self.multi = multi
#
#             modal1 = []
#             for i in range(self.multi):
#                 do = nn.Dropout(p=0.2)
#                 lin = nn.Linear(dim, dim)
#                 modal1.append(nn.Sequential(do, lin, nn.ReLU()))
#             self.modal1_layers = nn.ModuleList(modal1)
#
#             modal2 = []
#             for i in range(self.multi):
#                 do = nn.Dropout(p=0.2)
#                 lin = nn.Linear(dim, dim)
#                 modal2.append(nn.Sequential(do, lin, nn.ReLU()))
#             self.modal2_layers = nn.ModuleList(modal2)
#
#             modal3 = []
#             for i in range(self.multi):
#                 do = nn.Dropout(p=0.2)
#                 lin = nn.Linear(dim, dim)
#                 modal3.append(nn.Sequential(do, lin, nn.ReLU()))
#             self.modal3_layers = nn.ModuleList(modal3)
#
#         def forward(self, modal1_emb, modal2_emb, modal3_emb):
#             bs = modal1_emb.size(0)
#             x_mm = []
#             for i in range(self.multi):
#                 x_modal1 = self.modal1_layers[i](modal1_emb)
#                 x_modal2 = self.modal2_layers[i](modal2_emb)
#                 x_modal3 = self.modal3_layers[i](modal3_emb)
#                 x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
#             x_mm = torch.stack(x_mm, dim=1)
#             x_mm = x_mm.sum(1).view(bs, self.dim)
#             x_mm = torch.relu(x_mm)
#             return x_mm
#
#     def loss_func(self, output, target):
#         return self.bceloss(output, target)
#
#
# class TuckERLayer(nn.Module):
#     def __init__(self, dim, r_dim):
#         super(TuckERLayer, self).__init__()
#
#         self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
#         nn.init.xavier_uniform_(self.W.data)
#         self.bn0 = nn.BatchNorm1d(dim)
#         self.bn1 = nn.BatchNorm1d(dim)
#         self.input_drop = nn.Dropout(0.3)
#         self.hidden_drop = nn.Dropout(0.4)
#         self.out_drop = nn.Dropout(0.5)
#
#     def forward(self, e_embed, r_embed):
#         x = self.bn0(e_embed)
#         x = self.input_drop(x)
#         x = x.view(-1, 1, x.size(1))
#
#         r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
#         r = r.view(-1, x.size(2), x.size(2))
#         r = self.hidden_drop(r)
#
#         x = torch.bmm(x, r)
#         x = x.view(-1, x.size(2))
#         x = self.bn1(x)
#         x = self.out_drop(x)
#         return x
#
#
# class ModE(nn.Module):
#     def __init__(self, num_entity, num_relation, hidden_dim, gamma):
#         super(ModE, self).__init__()
#         self.num_entity = num_entity
#         self.num_relation = num_relation
#         self.hidden_dim = hidden_dim
#         self.epsilon = 2.0
#
#         self.gamma = nn.Parameter(
#             torch.Tensor([gamma]),
#             requires_grad=False
#         )
#
#         self.embedding_range = nn.Parameter(
#             torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
#             requires_grad=False
#         )
#
#         self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
#         nn.init.uniform_(
#             tensor=self.entity_embedding,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )
#
#         self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
#         nn.init.uniform_(
#             tensor=self.relation_embedding,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )
#
#     def forward(self, sample, batch_type=BatchType.SINGLE):
#         """
#         Given the indexes in `sample`, extract the corresponding embeddings,
#         and call func().
#
#         Args:
#             batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
#                 - SINGLE: positive samples in training, and all samples in validation / testing,
#                 - HEAD_BATCH: (?, r, t) tasks in training,
#                 - TAIL_BATCH: (h, r, ?) tasks in training.
#
#             sample: different format for different batch types.
#                 - SINGLE: tensor with shape [batch_size, 3]
#                 - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
#                     - positive_sample: tensor with shape [batch_size, 3]
#                     - negative_sample: tensor with shape [batch_size, negative_sample_size]
#         """
#         if batch_type == BatchType.SINGLE:
#             head = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=sample[:, 0]
#             ).unsqueeze(1)
#
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=sample[:, 1]
#             ).unsqueeze(1)
#
#             tail = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=sample[:, 2]
#             ).unsqueeze(1)
#
#         elif batch_type == BatchType.HEAD_BATCH:
#             tail_part, head_part = sample
#             batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
#
#             head = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=head_part.view(-1)
#             ).view(batch_size, negative_sample_size, -1)
#
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=tail_part[:, 1]
#             ).unsqueeze(1)
#
#             tail = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=tail_part[:, 2]
#             ).unsqueeze(1)
#
#         elif batch_type == BatchType.TAIL_BATCH:
#             head_part, tail_part = sample
#             batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
#
#             head = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=head_part[:, 0]
#             ).unsqueeze(1)
#
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=head_part[:, 1]
#             ).unsqueeze(1)
#
#             tail = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=tail_part.view(-1)
#             ).view(batch_size, negative_sample_size, -1)
#
#         else:
#             raise ValueError('batch_type %s not supported!'.format(batch_type))
#
#         # return scores
#         return self.func(head, relation, tail, batch_type)
#
#     def func(self, head, rel, tail, batch_type):
#         return self.gamma.item() - torch.norm(head * rel - tail, p=1, dim=2)
#
#
# class HAKE(nn.Module):
#     def __init__(self, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
#         super(HAKE, self).__init__()
#         self.num_entity = num_entity
#         self.num_relation = num_relation
#         self.hidden_dim = hidden_dim
#         self.epsilon = 2.0
#
#         self.gamma = nn.Parameter(
#             torch.Tensor([gamma]),
#             requires_grad=False
#         )
#
#         self.embedding_range = nn.Parameter(
#             torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
#             requires_grad=False
#         )
#
#         self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
#         nn.init.uniform_(
#             tensor=self.entity_embedding,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )
#
#         self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 3))
#         nn.init.uniform_(
#             tensor=self.relation_embedding,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )
#
#         nn.init.ones_(
#             tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
#         )
#
#         nn.init.zeros_(
#             tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
#         )
#
#         self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
#         self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))
#
#         self.pi = 3.14159262358979323846
#
#     def forward(self, sample, batch_type=BatchType.SINGLE):
#         """
#         Given the indexes in `sample`, extract the corresponding embeddings,
#         and call func().
#
#         Args:
#             batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
#                 - SINGLE: positive samples in training, and all samples in validation / testing,
#                 - HEAD_BATCH: (?, r, t) tasks in training,
#                 - TAIL_BATCH: (h, r, ?) tasks in training.
#
#             sample: different format for different batch types.
#                 - SINGLE: tensor with shape [batch_size, 3]
#                 - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
#                     - positive_sample: tensor with shape [batch_size, 3]
#                     - negative_sample: tensor with shape [batch_size, negative_sample_size]
#         """
#         if batch_type == BatchType.SINGLE:
#             head = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=sample[:, 0]
#             ).unsqueeze(1)
#
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=sample[:, 1]
#             ).unsqueeze(1)
#
#             tail = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=sample[:, 2]
#             ).unsqueeze(1)
#
#         elif batch_type == BatchType.HEAD_BATCH:
#             tail_part, head_part = sample
#             batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
#
#             head = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=head_part.view(-1)
#             ).view(batch_size, negative_sample_size, -1)
#
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=tail_part[:, 1]
#             ).unsqueeze(1)
#
#             tail = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=tail_part[:, 2]
#             ).unsqueeze(1)
#
#         elif batch_type == BatchType.TAIL_BATCH:
#             head_part, tail_part = sample
#             batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
#
#             head = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=head_part[:, 0]
#             ).unsqueeze(1)
#
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=head_part[:, 1]
#             ).unsqueeze(1)
#
#             tail = torch.index_select(
#                 self.entity_embedding,
#                 dim=0,
#                 index=tail_part.view(-1)
#             ).view(batch_size, negative_sample_size, -1)
#
#         else:
#             raise ValueError('batch_type %s not supported!'.format(batch_type))
#
#         # return scores
#         return self.func(head, relation, tail, batch_type)
#
#     def func(self, head, rel, tail, batch_type):
#         phase_head, mod_head = torch.chunk(head, 2, dim=2)
#         phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
#         phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)
#
#         phase_head = phase_head / (self.embedding_range.item() / self.pi)
#         phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
#         phase_tail = phase_tail / (self.embedding_range.item() / self.pi)
#
#         if batch_type == BatchType.HEAD_BATCH:
#             phase_score = phase_head + (phase_relation - phase_tail)
#         else:
#             phase_score = (phase_head + phase_relation) - phase_tail
#
#         mod_relation = torch.abs(mod_relation)
#         bias_relation = torch.clamp(bias_relation, max=1)
#         indicator = (bias_relation < -mod_relation)
#         bias_relation[indicator] = -mod_relation[indicator]
#
#         r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
#
#         phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
#         r_score = torch.norm(r_score, dim=2) * self.modulus_weight
#
#         return self.gamma.item() - (phase_score + r_score)
#
#
# class SubNet(nn.Module):
#     '''
#     The subnetwork that is used in LMF for image and node in the pre-fusion stage
#     '''
#
#     def __init__(self, in_size, hidden_size, dropout):
#         '''
#         Args:
#             in_size: input dimension
#             hidden_size: hidden layer dimension
#             dropout: dropout probability
#         Output:
#             (return value in forward) a tensor of shape (batch_size, hidden_size)
#         '''
#         super(SubNet, self).__init__()
#         self.norm = nn.BatchNorm1d(in_size)
#         self.drop = nn.Dropout(p=dropout)
#         self.linear_1 = nn.Linear(in_size, hidden_size)
#         self.linear_2 = nn.Linear(hidden_size, hidden_size)
#         self.linear_3 = nn.Linear(hidden_size, hidden_size)
#
#     def forward(self, x):
#         '''
#         Args:
#             x: tensor of shape (batch_size, in_size)
#         '''
#         normed = self.norm(x)
#         dropped = self.drop(normed)
#         y_1 = F.relu(self.linear_1(dropped))
#         y_2 = F.relu(self.linear_2(y_1))
#         y_3 = F.relu(self.linear_3(y_2))
#
#         return y_3
#
#
# # low-rank multimodal fusion
# class LMF(nn.Module):
#     '''
#     Low-rank Multimodal Fusion
#     '''
#
#     def __init__(self, input_dims, hidden_dims, desc_out, dropouts, output_dim, rank, use_softmax=False):
#         '''
#         Args:
#             input_dims - a length-3 tuple, contains (node_dim, image_dim, desc_dim)
#             hidden_dims - another length-3 tuple, hidden dims of the sub-networks
#             desc_out - int, specifying the resulting dimensions of the desc subnetwork
#             dropouts - a length-4 tuple, contains (node_dropout, image_dropout, desc_dropout, post_fusion_dropout)
#             output_dim - int, specifying the size of output
#             rank - int, specifying the size of rank in LMF
#         Output:
#             (return value in forward) a scalar value between -3 and 3
#         '''
#         super(LMF, self).__init__()
#
#         # dimensions are specified in the order of node, image and desc
#         self.node_in = input_dims[0]
#         self.image_in = input_dims[1]
#         self.desc_in = input_dims[2]
#
#         self.node_hidden = hidden_dims[0]
#         self.image_hidden = hidden_dims[1]
#         self.desc_hidden = hidden_dims[2]
#         self.desc_out = desc_out
#         self.output_dim = output_dim
#         self.rank = rank
#         self.use_softmax = use_softmax
#
#         self.node_prob = dropouts[0]
#         self.image_prob = dropouts[1]
#         self.desc_prob = dropouts[2]
#         self.post_fusion_prob = dropouts[3]
#
#         # define the pre-fusion subnetworks
#         self.node_subnet = SubNet(self.node_in, self.node_hidden, self.node_prob)
#         self.image_subnet = SubNet(self.image_in, self.image_hidden, self.image_prob)
#         self.desc_subnet = SubNet(self.desc_in, self.desc_hidden, self.desc_prob)
#
#         # define the post_fusion layers
#         self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
#         # self.post_fusion_layer_1 = nn.Linear((self.desc_out + 1) * (self.image_hidden + 1) * (self.node_hidden + 1), self.post_fusion_dim)
#         self.node_factor = Parameter(torch.Tensor(self.rank, self.node_hidden + 1, self.output_dim))
#         self.image_factor = Parameter(torch.Tensor(self.rank, self.image_hidden + 1, self.output_dim))
#         self.desc_factor = Parameter(torch.Tensor(self.rank, self.desc_hidden + 1, self.output_dim))
#         self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
#         self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
#
#         # init teh factors
#         xavier_normal(self.node_factor)
#         xavier_normal(self.image_factor)
#         xavier_normal(self.desc_factor)
#         xavier_normal(self.fusion_weights)
#         self.fusion_bias.data.fill_(0)
#
#     def forward(self, node_x, image_x, desc_x):
#         '''
#         Args:
#             node_x: tensor of shape (batch_size, node_in)
#             image_x: tensor of shape (batch_size, image_in)
#             desc_x: tensor of shape (batch_size, sequence_len, desc_in)
#         '''
#         node_h = self.node_subnet(node_x)
#         image_h = self.image_subnet(image_x)
#         desc_h = self.desc_subnet(desc_x)
#         batch_size = node_h.data.shape[0]
#
#         # next we perform low-rank multimodal fusion
#         # here is a more efficient implementation than the one the paper describes
#         # basically swapping the order of summation and elementwise product
#         if node_h.is_cuda:
#             DTYPE = torch.cuda.FloatTensor
#         else:
#             DTYPE = torch.FloatTensor
#
#         _node_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), node_h), dim=1)
#         _image_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), image_h), dim=1)
#         _desc_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), desc_h), dim=1)
#
#         fusion_node = torch.matmul(_node_h, self.node_factor)
#         fusion_image = torch.matmul(_image_h, self.image_factor)
#         fusion_desc = torch.matmul(_desc_h, self.desc_factor)
#         fusion_zy = fusion_node * fusion_image * fusion_desc
#
#         # output = torch.sum(fusion_zy, dim=0).squeeze()
#         # use linear transformation instead of simple summation, more flexibility
#         output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
#         output = output.view(-1, self.output_dim)
#         if self.use_softmax:
#             output = F.softmax(output)
#         return output


class ConvR(torch.nn.Module):
    def __init__(self, dim, num_entities, nodes):
        pass

    def forward(self, lhs, rel, candidate, score, start, end, queries):
        pass


class InteractE(torch.nn.Module):
    """
    Proposed method in the paper. Refer Section 6 of the paper for mode details

    Parameters
    ----------
    params:        	Hyperparameters of the model
    chequer_perm:   Reshaping to be used by the model

    Returns
    -------
    The InteractE model instance

    """

    def __init__(self, dim,nodes):
        super(InteractE, self).__init__()

        self.nodes = nodes.weight
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.embed_dim = dim
        self.perm = 1
        self.bn0 = torch.nn.BatchNorm2d(self.perm)
        self.k_h = 20
        self.k_w = 10
        self.num_filt = 96
        self.ker_sz = 9

        flat_sz_h = self.k_h
        flat_sz_w = 2 * self.k_w
        self.padding = 0

        self.bn1 = torch.nn.BatchNorm2d(self.num_filt * self.perm)
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt * self.perm

        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
        self.chequer_perm = self.get_chequer_perm()

        self.register_parameter('conv_filt', Parameter(torch.zeros(self.num_filt, 1, self.ker_sz, self.ker_sz)));

        self.sigmoid =nn.Sigmoid()
        nn.init.xavier_normal_(self.conv_filt)

    def get_chequer_perm(self):
        """
        Function to generate the chequer permutation required for InteractE model

        Parameters
        ----------

        Returns
        -------

        """
        ent_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])
        rel_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])

        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.k_h):
                for j in range(self.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim);
                            rel_idx += 1;
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim);
                            rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim);
                            rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                        else:
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim);
                            rel_idx += 1;

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to('cuda')
        return chequer_perm


    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def forward(self, lhs, rel, candidate, score, start, end, queries):
        sub_emb = lhs
        rel_emb = rel
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.perm, 2 * self.k_w, self.k_h))
        stack_inp = self.bn0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = self.circular_padding_chw(x, self.ker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.perm, 1, 1, 1), padding=self.padding, groups=self.perm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        to_score = self.nodes
        rhs = self.nodes[queries[:, 2]]
        if candidate:
            return  self.sigmoid(x @ to_score.transpose(0, 1)[start:end])
        if score:
            sco = torch.sum(x * (to_score[queries[:, 2]]), dim=1, keepdim=True)
            return self.sigmoid(sco)
        else:
            out = x @ to_score.transpose(0, 1)
            out = self.sigmoid(out)

            return (out, (
                lhs,
                rel,
                rhs
            )
                    )


class TransformerEncoderModel(nn.Module):
    def __init__(self, dim, hidden_size, num_layers, num_heads):
        super(TransformerEncoderModel, self).__init__()

        # Define the Transformer Encoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=hidden_size,
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(dim, dim//4)

    def forward(self, input_data):
        # input_data: (seq_len, batch_size, input_size)

        # Transformer Encoder expects input in the shape (seq_len, batch_size, input_size)
        transformer_output = self.transformer_encoder(input_data)

        pooled_output = transformer_output.mean(dim=0)

        output = self.fc(pooled_output)

        return output








class ConvE(torch.nn.Module):
    def __init__(self, dim, num_entities, nodes):
        super(ConvE, self).__init__()
        self.nodes = nodes.weight
        self.n_node = self.nodes.shape[0]
        self.to_score =nn.Parameter(torch.Tensor(self.n_node, dim), requires_grad=True)
        nn.init.xavier_uniform_(self.to_score)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = 16
        self.emb_dim2 = dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        # self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(28800, dim)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, lhs,rel,rhs,to_score, candidate, score, start, end, queries):
        e1_embedded = lhs.view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = rel.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        # print(stacked_inputs.shape)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        to_score = to_score
        rhs = rhs
        if candidate:
            return x @ to_score.transpose(0, 1)[start:end]
        if score:
            sco = torch.sum(x * (to_score[queries[:, 2]]), dim=1, keepdim=True)
            return sco
        else:
            out = x @ to_score.transpose(0, 1)

            return (out, (
                lhs,
                rel,
                rhs
            )
                    )
