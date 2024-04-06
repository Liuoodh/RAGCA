from base_models import *



class KBCModel(nn.Module, ABC):
    @abstractmethod
    def candidates_score(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    # 二分类式训练
    def forward_bpr(self, batch):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        self.eval()
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]

                    scores = self.candidates_score(c_begin, chunk_size, these_queries)
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        # 当前满足 （head,relation，tail） 的tail组成的列表
                        filter_out = filters[(query[0].item(), query[1].item())]
                        # 真实tail id 放进去
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            # 真实分数置为-1e6
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    # 一个batch中大于等于当前预测得分的尾实体的总数（过滤掉存在的list） （+1得到排名）
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class RAGCA(KBCModel):
    def __init__(self, sizes, n_node, n_rel, dim, n_neighbor, context_weight,adj, img_info, desc_info, node_info=None,
                 rel_desc_info=None):
        super(RAGCA, self).__init__()
        self.sizes = sizes
        self.n_node = n_node
        self.n_rel = n_rel
        self.dim = dim
        self.tem=16
        self.context_weight = context_weight

        self.adj_indices, self.adj_values = adj[0].to('cuda'), adj[1].to('cuda')

        self.n_neighbor = n_neighbor



        if img_info is not None:
            self.img_info = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open(img_info, 'rb'))), freeze=False)
        if desc_info is not None:
            self.desc_info = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open(desc_info, 'rb'))), freeze=False)
        if node_info is not None:
            # self.node_info = nn.Embedding.from_pretrained(
            #     torch.from_numpy(pickle.load(open(node_info, 'rb'))), freeze=False)
            self.node_info = nn.Embedding(n_node, dim, sparse=True)
            nn.init.xavier_uniform_(self.node_info.weight)
            # print(n_node)
            # print(self.node_info.weight.shape)
        else:
            self.node_info = nn.Embedding(n_node, dim, sparse=True)
            nn.init.xavier_uniform_(self.node_info.weight)

        if rel_desc_info is not None:
            self.rel_info = nn.Embedding.from_pretrained(
                torch.vstack((torch.from_numpy(pickle.load(open(rel_desc_info, 'rb'))),
                              torch.from_numpy(pickle.load(open(rel_desc_info, 'rb')))))
                , freeze=False)
            # self.rel_info = nn.Embedding(n_rel * 2, dim, sparse=True)
            # nn.init.xavier_uniform_(self.rel_info.weight)
            # print(self.rel_info.weight.shape)
        else:
            self.rel_info = nn.Embedding(n_rel * 2, dim, sparse=True)
            nn.init.xavier_uniform_(self.rel_info.weight)

        self.weight_i=nn.Parameter(torch.Tensor(dim, dim), requires_grad=True)

        nn.init.xavier_uniform_(self.weight_i)

        self.weight_d = nn.Parameter(torch.Tensor(dim, dim), requires_grad=True)

        nn.init.xavier_uniform_(self.weight_d)

        self.weight_n = nn.Parameter(torch.Tensor(dim, dim), requires_grad=True)

        nn.init.xavier_uniform_(self.weight_n)

        self.rel_guide_weight = nn.Parameter(torch.Tensor(dim, 3), requires_grad=True)
        nn.init.xavier_uniform_(self.rel_guide_weight)

        self.to_score = nn.Parameter(torch.Tensor(self.n_node, dim), requires_grad=True)
        nn.init.xavier_uniform_(self.to_score)
        #
        # self._score = ConvE(dim, n_node, self.node_info)
        # # self._score = InteractE(dim, self.node_info)
        # self._score = TuckER(self.node_info,dim)
        self._score = ComplEx(dim//2,self.node_info)

        self._score_context = ConvE(dim, n_node, self.node_info)
        # self._score_context = ComplEx(dim//2,self.node_info)

        self.context_enhancement_net=ContextE(dim)

    def forward(self, x, candidate=False, score=False, chunk_begin=-1, chunk_size=-1):
        lhs_node = self.node_info.weight[x[:, 0]]
        lhs_desc = self.desc_info.weight[x[:, 0]]
        lhs_img = self.img_info.weight[x[:, 0]]

        rel = self.rel_info.weight[x[:, 1]]

        desc,img = self.relation_attention_guide_filter(lhs_node, rel, lhs_desc, lhs_img)

        # print(desc.shape)
        # print(rel.shape)

        trans_n = torch.transpose(lhs_node.unsqueeze(2), 1, 2)

        M_d = torch.bmm(desc.unsqueeze(2).to(torch.float32),trans_n)

        M_i = torch.bmm(img.unsqueeze(2).to(torch.float32),trans_n)


        # nx1xd
        a_n = torch.transpose(F.softmax(rel.mm(self.weight_n),dim=1).unsqueeze(2),1,2)

        a_d=F.softmax(rel.mm(self.weight_d),dim=1)
        # nxdxd
        mask_d=torch.bmm(a_d.unsqueeze(2),a_n)

        a_i =F.softmax(rel.mm(self.weight_i),dim=1)

        mask_i = torch.bmm(a_i.unsqueeze(2), a_n)


        desc = torch.bmm(M_d*mask_d, rel.unsqueeze(2)).squeeze(2)

        img = torch.bmm(M_i*mask_i, rel.unsqueeze(2)).squeeze(2)


        guide_weight = F.softmax(rel.mm(self.rel_guide_weight), dim=1)
        masks = (torch.unsqueeze(guide_weight[:, 0], dim=1),torch.unsqueeze(guide_weight[:, 1], dim=1),torch.unsqueeze(guide_weight[:, 2], dim=1))
        fusion_info = (masks[0]*lhs_node+masks[1]*desc+masks[2]*img).to(torch.float32)

        # batch_size x k x dim

        context_rel = self.get_context_rel(self.rel_info.weight,x)
        context_node = self.get_context_nodes(self.node_info.weight,x)
        context_desc = self.get_context_nodes(self.desc_info.weight,x)
        context_img = self.get_context_nodes(self.img_info.weight,x)

        context_input = torch.stack((context_node, context_desc, context_img, context_rel), dim=1)
        context_info = self.context_enhancement_net(context_input,lhs_node,rel)




        rhs_node = self.node_info.weight[x[:, 2]]


        assert not torch.isnan(fusion_info).any()



        pred_node = self._score(fusion_info, rel,rhs = rhs_node,to_score=self.node_info.weight,candidate=candidate, score=score, start=chunk_begin,
                            end=chunk_begin + chunk_size, queries=x)



        pred_context = self._score_context(context_info,rel,rhs = rhs_node,to_score=self.node_info.weight,candidate=candidate, score=score, start=chunk_begin,
                            end=chunk_begin + chunk_size, queries=x)


        if candidate or score:
            pred = (1-self.context_weight)*pred_node + self.context_weight*pred_context
        else:
            pred = (*pred_node,*pred_context)


        return pred


    # 关系注意力引导过滤
    def relation_attention_guide_filter(self,lhs_node,rel,desc_info,img_info):
        node = F.normalize(lhs_node, p=2, dim=1)
        rel = F.normalize(rel, p=2, dim=1)
        desc = F.normalize(desc_info, p=2, dim=1)
        img = F.normalize(img_info, p=2, dim=1)
        cosine_node_desc = torch.unsqueeze(F.cosine_similarity(node,desc,dim=1),dim=1)
        cosine_node_img = torch.unsqueeze(F.cosine_similarity(node, img,dim=1),dim=1)
        # cosine_rel_desc = torch.unsqueeze(F.cosine_similarity(rel, desc,dim=1),dim=1)
        # cosine_rel_img = torch.unsqueeze(F.cosine_similarity(rel, img, dim=1),dim=1)
        # weight = F.softmax(rel,dim=1)
        # filter_desc = (cosine_node_desc+cosine_rel_desc)*desc_info*weight
        # filter_img  = (cosine_node_img+cosine_rel_img)*img_info*weight
        filter_desc = (cosine_node_desc)*desc_info
        filter_img  = (cosine_node_img)*img_info
        return filter_desc,filter_img

    # 邻居节点融合
    def fusion_neighbor(self, lhs_node, rel, desc_info, img_info):
        filter_desc,filter_img =self.relation_attention_guide_filter(lhs_node, rel, desc_info, img_info)
        return lhs_node+filter_desc+filter_img

    # 得到当前节点周围 K 个 邻居的嵌入 先把当前节点所有相关的 关系嵌入和当前节点一起输入 再输出 然后把输出和邻居节点们一起输入 再输出
    # 三种模态 运算三次 然后再融合 随机加入mask 隐藏源实体  防止上下文信息不足 加入 重构损失 重建源实体
    def get_context_rel(self,rels, x):
        # 源实体
        # with torch.no_grad():
        batch_size = len(x)
        relevant_rels = torch.zeros((batch_size, self.n_neighbor, self.dim)).to('cuda')  # 创建结果张量，形状为 batch_size x k x dim

        # 获取x中所有头节点对应的关系的索引
        head_nodes = x[:, 0].to('cuda')  # 提取所有头节点

        head_indices = torch.nonzero((self.adj_indices[:, 0].unsqueeze(0) == head_nodes.unsqueeze(1))).to('cuda')

        # 获取每个头节点对应的关系数量 维度为batch_size 值为每个节点拥有的关系数量
        head_counts = torch.bincount(head_indices[:, 0], minlength=batch_size).to('cuda')

        # 找到关系数量小于k的头节点的id
        flag = head_counts < self.n_neighbor

        # 获取关系数量小于k的头节点的索引
        insufficient_head_indices = torch.nonzero(flag).squeeze().to('cuda')


        insufficient_head_indices_len = insufficient_head_indices.shape[
            0] if insufficient_head_indices.numel() != 0 else 0

        sufficient_head_indices = torch.nonzero(~flag).squeeze().to('cuda')

        sufficient_head_indices_len = sufficient_head_indices.shape[0] if sufficient_head_indices.numel() != 0 else 0


        # len(insufficient_head_indices) 表示有几个关系数量小于k的头节点
        # 以下循环运算 batch_size 越小 运算越快
        if insufficient_head_indices_len > 0:
            for idx in insufficient_head_indices:
                mask = head_indices[:, 0] == idx
                rel_indices=self.adj_values[head_indices[mask][:, 1]]


                # rel_indices=torch.masked_select(rel_indices,rel_indices !=x[idx][1])

                insufficient_head_relations = rels[rel_indices]
                # 创建全零嵌入矩阵
                zero_embeddings = torch.zeros((self.n_neighbor - insufficient_head_relations.shape[0], rels.shape[1])).to('cuda')

                # 将嵌入向量存储在结果中
                relevant_rels[idx] = torch.cat((insufficient_head_relations, zero_embeddings))


        if sufficient_head_indices_len > 0:
            for idx in sufficient_head_indices:
                mask = head_indices[:, 0] == idx
                rel_indices=self.adj_values[head_indices[mask][:self.n_neighbor, 1]]
                # rel_indices=torch.masked_select(rel_indices,rel_indices != x[idx][1])
                sufficient_head_relations = rels[rel_indices]
                if sufficient_head_relations.shape[0] < self.n_neighbor:

                    zero_embeddings = torch.zeros((self.n_neighbor - sufficient_head_relations.shape[0], rels.shape[1])).to(
                        'cuda')
                    relevant_rels[idx] = torch.cat((sufficient_head_relations, zero_embeddings))

                else:
                    relevant_rels[idx] = sufficient_head_relations


        return relevant_rels

    def get_context_nodes(self, nodes, x):
        # 源实体
        # with torch.no_grad():
        batch_size = len(x)

        relevant_nodes = torch.zeros((batch_size, self.n_neighbor, self.dim)).to('cuda')

        # 获取x中所有头节点对应的关系的索引
        head_nodes = x[:, 0].to('cuda')  # 提取所有头节点

        head_indices = torch.nonzero((self.adj_indices[:, 0].unsqueeze(0) == head_nodes.unsqueeze(1))).to('cuda')

        # 获取每个头节点对应的关系数量 维度为batch_size 值为每个节点拥有的关系数量
        head_counts = torch.bincount(head_indices[:, 0], minlength=batch_size).to('cuda')

        tail_indices = self.adj_indices[:, 1].to('cuda')
        # 找到关系数量小于k的头节点的id
        flag = head_counts < self.n_neighbor

        # 获取关系数量小于k的头节点的索引
        insufficient_head_indices = torch.nonzero(flag).squeeze().to('cuda')

        insufficient_head_indices_len = insufficient_head_indices.shape[
            0] if insufficient_head_indices.numel() != 0 else 0

        sufficient_head_indices = torch.nonzero(~flag).squeeze().to('cuda')

        sufficient_head_indices_len = sufficient_head_indices.shape[0] if sufficient_head_indices.numel() != 0 else 0

        # len(insufficient_head_indices) 表示有几个关系数量小于k的头节点
        # 以下循环运算 batch_size 越小 运算越快
        if insufficient_head_indices_len > 0:
            for idx in insufficient_head_indices:
                mask = head_indices[:, 0] == idx


                node_indices = tail_indices[head_indices[mask][:, 1]]

                # node_indices = torch.masked_select(node_indices, node_indices != x[idx][2])


                insufficient_head_nodes = nodes[node_indices]

                # 创建全零嵌入矩阵
                zero_embeddings = torch.zeros((self.n_neighbor - insufficient_head_nodes.shape[0], nodes.shape[1])).to('cuda')



                relevant_nodes[idx] = torch.cat((insufficient_head_nodes, zero_embeddings))

        if sufficient_head_indices_len > 0:
            for idx in sufficient_head_indices:
                mask = head_indices[:, 0] == idx
                node_indices = tail_indices[head_indices[mask][:self.n_neighbor, 1]]
                # node_indices = torch.masked_select(node_indices, node_indices != x[idx][2])
                sufficient_head_nodes = nodes[node_indices]
                if sufficient_head_nodes.shape[0] < self.n_neighbor:
                    zero_embeddings = torch.zeros((self.n_neighbor - sufficient_head_nodes.shape[0], nodes.shape[1])).to(
                        'cuda')
                    relevant_nodes[idx] = torch.cat((sufficient_head_nodes, zero_embeddings))

                else:
                    relevant_nodes[idx] = sufficient_head_nodes

        return relevant_nodes

    def candidates_score(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):

        return self.forward(queries, candidate=True, chunk_begin=chunk_begin, chunk_size=chunk_size)

    def score(self, x: torch.Tensor):

        return self.forward(x, score=True)

    def forward_bpr(self, batch):
        # pos_scores = self.score(pos)
        # neg_scores = self.score(neg)
        scores = self.score(batch)
        # delta = pos_scores - neg_scores
        # fac = self.get_factor(torch.cat((pos, neg), dim=0))
        return scores



class ContextE(nn.Module):
    def __init__(self, dim):
        super(ContextE, self).__init__()

        self.conv1 = nn.Conv2d(4, 4, (3, 3), 1, 0, bias=True)


        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attention_fc = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU(),
            nn.Linear(1, 4),
            nn.Sigmoid()
        )

        self.bn0 = torch.nn.BatchNorm2d(4)

        self.bn1 = torch.nn.BatchNorm2d(4)

        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.inp_drop = torch.nn.Dropout(0.2)

        self.hidden_drop = torch.nn.Dropout(0.3)

        self.feature_map_drop = torch.nn.Dropout2d(0.2)

        # 全连接层
        self.fc = nn.Linear(97920, dim)

    def forward(self, x,node,rel):
        a = torch.mean(node, dim=1, keepdim=True)
        b = torch.mean(rel, dim=1, keepdim=True)
        guide = torch.concat((a,b),dim=1)
        channel_attention = self.attention_fc(guide)
        channel_attention = torch.unsqueeze(torch.unsqueeze(channel_attention, -1), -1)

        x = self.conv1(x)

        x = x*channel_attention
        x = self.bn1(x)

        x = self.feature_map_drop(x)
        # x = self.pool(x)

        # 第二层卷积 + 池化
        x = F.relu(x)

        # x = self.pool(x)

        # 将张量展平为一维
        x = x.view(x.shape[0], -1)

        # 全连接层
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


