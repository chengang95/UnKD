"""
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import math

import world
import torch
import numpy as np
from torch import nn
from dataloader import BasicDataset
from torch.autograd import Variable
import utils
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg, weights=None):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def pair_score(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            Sigmoid(Score_pos - Score_neg)
        """
        raise NotImplementedError

class DistillEmbedding(BasicModel):
    '''
        student's embedding is not total free
    '''
    def __init__(self, *args):
        super(DistillEmbedding, self).__init__()

    @property
    def embedding_user(self):
        raise NotImplementedError
    @property
    def embedding_item(self):
        raise NotImplementedError


class Embedding_wrapper:
    def __init__(self, num_embeddings, embedding_dim):
        self.num = num_embeddings
        self.dim = embedding_dim
        self.weight = None

    def __call__(self,
                 index : torch.Tensor):
        if not isinstance(index, torch.LongTensor):
            index = index.long()
        if self.weight is not None:
            return self.weight[index]
        else:
            raise TypeError("haven't update embedding")

    def pass_weight(self, weight):
        try:
            assert len(weight.shape)
            assert weight.shape[0] == self.num
            assert weight.shape[1] == self.dim
            self.weight = weight
        except AssertionError:
            raise AssertionError(f"weight your pass is wrong! \n expect {self.num}X{self.dim}, but got {weight.shapet}")

    def __repr__(self):
        return f"Emb({self.num} X {self.dim})"


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset,
                 fix:bool = False):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset=dataset
        self.__init_weight()
        self.fix=fix

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma





class BPRMF(PairWiseModel):
    '''
    PD/PDA
    PDG/PDG-A
    '''

    def __init__( self,config:dict,
                 dataset:BasicDataset,
                 fix:bool = False,):
        super(BPRMF, self).__init__()
        self.config=config
        self.dataset=dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.fix=fix
        self.init_weights()


    def init_weights(self):

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        self.f = nn.Sigmoid()
        self.felu=nn.ReLU()

    def getUsersRating(self, users):
        users_emb = self.embedding_user.weight[users]
        items = torch.Tensor(range(self.dataset.m_items)).long().cuda()
        items_emb = self.embedding_item.weight[items]
        rating = torch.matmul(users_emb, items_emb.t())
        #rating = self.felu(rating) + 1
        return rating


    def create_bpr_loss(self, users, pos_items,
                                        neg_items):  # this global does not refer to global popularity, just a name
        users_emb = self.embedding_user.weight[users]
        pos_emb = self.embedding_item.weight[pos_items]
        neg_emb = self.embedding_item.weight[neg_items]
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        #pos_scores = self.felu(pos_scores) + 1
        #neg_scores = self.felu(neg_scores) + 1

        maxi = torch.log(self.f(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.mean(torch.neg(maxi))

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2))/float(len(users))
        return mf_loss, reg_loss



    def bpr_loss(self, users, pos_items,neg_items,weights=None):
        return  self.create_bpr_loss(users, pos_items,neg_items)

    def forward(self, users, items):
        """
        without sigmoid
        """
        users_emb=self.embedding_user.weight[users]
        items_emb=self.embedding_item.weight[items]
        rating = torch.sum(users_emb*items_emb,dim=1)
        #rating = self.felu(rating) + 1
        return rating

class PDALightGCN(LightGCN):
    '''
    PD/PDA
    PDG/PDG-A
    '''

    def __init__( self,config:dict,
                 dataset:BasicDataset,
                 fix:bool = False):
        LightGCN.__init__(self, config, dataset)

        self.dataset.item_weight.requires_grad = False
        print(self.dataset.item_weight)


    def bpr_loss(self, users, pos_items,neg_items):  # this global does not refer to global popularity, just a name


        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        pos_scores = torch.relu(pos_scores) + 1
        neg_scores = torch.relu(neg_scores) + 1
        pos_weight = self.dataset.item_weight[pos_items].cuda()
        neg_weight = self.dataset.item_weight[neg_items].cuda()

        maxi = torch.log(torch.sigmoid(pos_weight * pos_scores - neg_weight * neg_scores) + 1e-10)

        mf_loss = torch.mean(torch.neg(maxi))

        return mf_loss, reg_loss



class PDA(PairWiseModel):
    '''
    PD/PDA
    PDG/PDG-A
    '''

    def __init__( self,config:dict,
                 dataset:BasicDataset,
                 fix:bool = False,):
        super(PDA, self).__init__()
        self.config=config
        self.dataset=dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.fix=fix
        self.init_weights()

        self.dataset.item_weight.requires_grad = False


    def init_weights(self):

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        self.f = nn.Sigmoid()
        self.felu=nn.ReLU()

    def getUsersRating(self, users):
        users_emb = self.embedding_user.weight[users]
        items = torch.Tensor(range(self.dataset.m_items)).long().cuda()
        items_emb = self.embedding_item.weight[items]
        rating = torch.matmul(users_emb, items_emb.t())
        #rating = self.felu(rating) + 1
        # w = self.dataset.item_weight[items].cuda()
        # rating=w*rating
        return rating


    def create_bpr_loss(self, users, pos_items,
                                        neg_items):  # this global does not refer to global popularity, just a name
        users_emb = self.embedding_user.weight[users]
        pos_emb = self.embedding_item.weight[pos_items]
        neg_emb = self.embedding_item.weight[neg_items]
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        pos_scores = self.felu(pos_scores) + 1
        neg_scores = self.felu(neg_scores) + 1
        pos_weight=self.dataset.item_weight[pos_items].cuda()
        neg_weight = self.dataset.item_weight[neg_items].cuda()

        maxi = torch.log(self.f(pos_weight*pos_scores - neg_weight*neg_scores) + 1e-10)

        mf_loss = torch.mean(torch.neg(maxi))

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2))/float(len(users))
        return mf_loss, reg_loss



    def bpr_loss(self, users, pos_items,neg_items,weights=None):
        return  self.create_bpr_loss(users, pos_items,neg_items)

    def forward(self, users, items):
        """
        without sigmoid
        """
        users_emb=self.embedding_user.weight[users]
        items_emb=self.embedding_item.weight[items]
        rating = torch.sum(users_emb*items_emb,dim=1)
        # rating = self.felu(rating) + 1
        # w=self.dataset.itwm_weight[items].cuda()
        # rating=w*rating
        return rating


class OneLinear(nn.Module):
    """
    linear model: r
    """

    def __init__(self, n,k):
        super().__init__()
        self.fc1=nn.Linear(n,int(n/2))
        self.fc2 = nn.Linear(int(n/2), k)

    def forward(self, x):
        x=self.fc1(x)
        x=self.fc2(x)
        return x



class Expert(nn.Module):
    def __init__(self, dims):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential()
        L1=nn.Linear(dims[0], dims[1])
        L2=nn.Linear(dims[1], dims[2])
        self.mlp.add_module("L1",L1)
        self.mlp.add_module("L2", L2)
        self.mlp.add_module("r", nn.ReLU())

    def forward(self, x):
        return self.mlp(x)


class BPRMFExpert(LightGCN):
        def __init__(self,
                     config: dict,
                     dataset: BasicDataset,
                     teacher_model: LightGCN,
                     ):
            LightGCN.__init__(self, config, dataset)
            self.config = config
            self.dataset = dataset
            self.tea = teacher_model
            self.tea.fix = True
            self.de_weight = config['de_weight']

            # Expert Configuration
            self.num_experts = self.config["num_expert"]
            self.latent_dim_tea = self.tea.latent_dim
            expert_dims = [self.latent_dim, (self.latent_dim_tea + self.latent_dim) // 2, self.latent_dim_tea]

            ## for self-distillation
            if self.tea.latent_dim == self.latent_dim:
                expert_dims = [self.latent_dim, self.latent_dim_tea // 2, self.latent_dim_tea]

            self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
            self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

            self.user_selection_net = nn.Sequential(nn.Linear(self.latent_dim_tea, self.num_experts), nn.Softmax(dim=1))
            self.item_selection_net = nn.Sequential(nn.Linear(self.latent_dim_tea, self.num_experts), nn.Softmax(dim=1))

            # num_experts_params = count_parameters(self.user_experts) + count_parameters(self.item_experts)
            # num_gates_params = count_parameters(self.user_selection_net) + count_parameters(self.item_selection_net)

            self.sm = nn.Softmax(dim=1)

            self.T = 10

        def get_DE_loss(self, batch_entity, is_user=True):

            if is_user:
                s = self.embedding_user.weight[batch_entity]
                t = self.tea.embedding_user.weight[batch_entity]

                experts = self.user_experts
                selection_net = self.user_selection_net

            else:
                s = self.embedding_item.weight[batch_entity]
                t = self.tea.embedding_item.weight[batch_entity]

                experts = self.item_experts
                selection_net = self.item_selection_net

            selection_dist = selection_net(t)  # batch_size x num_experts

            if self.num_experts == 1:
                selection_result = 1.
            else:
                # Expert Selection
                g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).cuda()
                eps = 1e-10  # for numerical stability
                selection_dist = selection_dist + eps
                selection_dist = self.sm((selection_dist.log() + g) / self.T)
                selection_dist = torch.unsqueeze(selection_dist, 1)  # batch_size x 1 x num_experts
                selection_result = selection_dist.repeat(1, self.latent_dim_tea,
                                                         1)  # batch_size x teacher_dims x num_experts
            expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)]  # s -> t
            expert_outputs = torch.cat(expert_outputs, -1)  # batch_size x teacher_dims x num_experts
            expert_outputs = expert_outputs * selection_result  # batch_size x teacher_dims x num_experts
            expert_outputs = expert_outputs.sum(-1)  # batch_size x teacher_dims
            DE_loss = torch.mean(((t - expert_outputs) ** 2).sum(-1))
            return DE_loss



def sim(A, B, is_inner=False):
    if not is_inner:
        denom_A = 1 / (A ** 2).sum(1, keepdim=True).sqrt()
        denom_B = 1 / (B.T ** 2).sum(0, keepdim=True).sqrt()

        sim_mat = torch.mm(A, B.T) * denom_A * denom_B
    else:
        sim_mat = torch.mm(A, B.T)

    return sim_mat

class f(nn.Module):
	def __init__(self, dims):
		super(f, self).__init__()

		self.net = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2]))

	def forward(self, x):
		return self.net(x)

class HTD(LightGCN):
    def __init__(self, config:dict,
                 dataset:BasicDataset, teacher_model, K=30, choice='second'):

        LightGCN.__init__(self, config, dataset)

        self.config = config
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.student_dim = self.config['latent_dim_rec']
        user_emb_teacher=teacher_model.embedding_user.weight
        item_emb_teacher=teacher_model.embedding_item.weight
        # Teacher
        self.user_emb_teacher = nn.Embedding.from_pretrained(user_emb_teacher)
        self.item_emb_teacher = nn.Embedding.from_pretrained(item_emb_teacher)

        self.user_emb_teacher.weight.requires_grad = False
        self.item_emb_teacher.weight.requires_grad = False

        self.teacher_dim = self.user_emb_teacher.weight.size(1)

        # Group Assignment related parameters
        self.K = K
        F_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]

        self.user_f = nn.ModuleList([f(F_dims) for i in range(self.K)])
        self.item_f = nn.ModuleList([f(F_dims) for i in range(self.K)])

        self.user_v = nn.Sequential(nn.Linear(self.teacher_dim, K), nn.Softmax(dim=1))
        self.item_v = nn.Sequential(nn.Linear(self.teacher_dim, K), nn.Softmax(dim=1))

        self.sm = nn.Softmax(dim=1)
        self.T = 0.1

        # Group-Level topology design choices
        self.choice = choice

    def get_group_result(self, batch_entity, is_user=True):
        with torch.no_grad():
            if is_user:
                t = self.user_emb_teacher(batch_entity)
                v = self.user_v
            else:
                t = self.item_emb_teacher(batch_entity)
                v = self.item_v

            z = v(t).max(-1)[1]
            if not is_user:
                z = z + self.K

            return z

    # For Adaptive Group Assignment
    def get_GA_loss(self, batch_entity, is_user=True):

        if is_user:
            s = self.embedding_user(batch_entity)
            t = self.user_emb_teacher(batch_entity)

            f = self.user_f
            v = self.user_v
        else:
            s = self.embedding_item(batch_entity)
            t = self.item_emb_teacher(batch_entity)

            f = self.item_f
            v = self.item_v

        alpha = v(t)
        g = torch.distributions.Gumbel(0, 1).sample(alpha.size()).cuda()
        alpha = alpha + 1e-10
        z = self.sm((alpha.log() + g) / self.T)

        z = torch.unsqueeze(z, 1)
        z = z.repeat(1, self.teacher_dim, 1)

        f_hat = [f[i](s).unsqueeze(-1) for i in range(self.K)]
        f_hat = torch.cat(f_hat, -1)
        f_hat = f_hat * z
        f_hat = f_hat.sum(2)

        GA_loss = ((t - f_hat) ** 2).sum(-1).sum()

        return GA_loss

    def get_TD_loss(self, batch_user, batch_item):
        if self.choice == 'first':
            return self.get_TD_loss1(batch_user, batch_item)
        else:
            return self.get_TD_loss2(batch_user, batch_item)

    # Topology Distillation Loss (with Group(P,P))
    def get_TD_loss1(self, batch_user, batch_item):

        s = torch.cat([self.user_emb(batch_user), self.item_emb(batch_item)], 0)
        t = torch.cat([self.user_emb_teacher(batch_user), self.item_emb_teacher(batch_item)], 0)
        z = torch.cat(
            [self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
        G_set = z.unique()
        Z = F.one_hot(z).float()

        # Compute Prototype
        with torch.no_grad():
            tmp = Z.T
            tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
            P_s = tmp.mm(s)[G_set]
            P_t = tmp.mm(t)[G_set]

        # entity_level topology
        entity_mask = Z.mm(Z.T)

        t_sim_tmp = sim(t, t) * entity_mask
        t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]

        s_sim_dist = sim(s, s) * entity_mask
        s_sim_dist = s_sim_dist[t_sim_tmp > 0.]

        # # Group_level topology
        t_proto_dist = sim(P_t, P_t).view(-1)
        s_proto_dist = sim(P_s, P_s).view(-1)

        total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

        return total_loss

    # Topology Distillation Loss (with Group(P,e))
    def get_TD_loss2(self, batch_user, batch_item):

        s = torch.cat([self.embedding_user(batch_user), self.embedding_item(batch_item)], 0)
        t = torch.cat([self.user_emb_teacher(batch_user), self.item_emb_teacher(batch_item)], 0)
        z = torch.cat(
            [self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
        G_set = z.unique()
        Z = F.one_hot(z).float()

        # Compute Prototype
        with torch.no_grad():
            tmp = Z.T
            tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
            P_s = tmp.mm(s)[G_set]
            P_t = tmp.mm(t)[G_set]

        # entity_level topology
        entity_mask = Z.mm(Z.T)

        t_sim_tmp = sim(t, t) * entity_mask
        t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]

        s_sim_dist = sim(s, s) * entity_mask
        s_sim_dist = s_sim_dist[t_sim_tmp > 0.]

        # Group_level topology
        t_proto_dist = (sim(P_t, t) * (1 - Z.T)[G_set]).view(-1)
        s_proto_dist = (sim(P_s, s) * (1 - Z.T)[G_set]).view(-1)

        # t_proto_dist = sim(P_t, t).view(-1)
        # s_proto_dist = sim(P_s, s).view(-1)

        total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

        return total_loss
