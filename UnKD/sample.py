import math

import world
import torch
import multiprocessing
import numpy as np
from torch.nn.functional import softplus
from time import time
from utils import shapes, combinations, timer
from world import cprint
from model import PairWiseModel,BasicModel
from dataloader import BasicDataset, Loader
from torch.nn import Softmax, Sigmoid
import torch.nn.functional as F
import utils
import random

# try:
#     from cppimport import imp_from_filepath
#     from os.path import join, dirname
#     path = join(dirname(__file__), "sources/sampling.cpp")
#     sampling = imp_from_filepath(path)
#     sampling.seed(world.SEED)
#     sample_ext = True
# except:
#     world.cprint("Cpp ext not loaded")
#     sample_ext = False


sample_ext = False
ALLPOS = None
# ----------------------------------------------------------------------------
# distill


def userAndMatrix(batch_users, batch_items, model):
    """cal scores between user vector and item matrix

    Args:
        batch_users (tensor): vector (batch_size)
        batch_items (tensor): matrix (batch_size, dim_item)
        model (PairWiseModel):

    Returns:
        tensor: scores, shape like batch_items
    """
    dim_item = batch_items.shape[-1]
    vector_user = batch_users.repeat((dim_item, 1)).t().reshape((-1, ))
    vector_item = batch_items.reshape((-1, ))
    return model(vector_user, vector_item).reshape((-1, dim_item))


class DistillSample:
    def __init__(self,
                 dataset: BasicDataset,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns_k: int,
                 method: int = 3,
                 beta=world.beta):
        """
            method 1 for convex combination
            method 2 for random indicator
            method 3 for simplified method 2
        """
        self.beta = beta
        self.W = torch.Tensor([world.p0])
        self.dataset = dataset
        self.student = student
        self.teacher = teacher
        # self.methods = {
        #     'combine' : self.convex_combine, # not yet
        #     'indicator' : self.random_indicator,
        #     'simple' : self.max_min,
        #     'weight' : self.weight_pair,
        # }
        self.method = 'combine'
        self.Sample = self.convex_combine
        cprint(f"Using {self.method}")
        self.dns_k = dns_k
        self.soft = Softmax(dim=1)
        # self._generateTopK()

    def PerSample(self, batch=None):

        return Sample_DNS_python(self.dataset, self.dns_k)

    def _generateTopK(self, batch_size=256):
        if self.RANK is None:
            with torch.no_grad():
                self.RANK = torch.zeros((self.dataset.n_users, self.topk))
                for user in range(0, self.dataset.n_users, batch_size):
                    end = min(user + batch_size, self.dataset.n_users)
                    scores = self.teacher.getUsersRating(
                        torch.arange(user, end))
                    pos_item = self.dataset.getUserPosItems(
                        np.arange(user, end))

                    # -----
                    exclude_user, exclude_item = [], []
                    for i, items in enumerate(pos_item):
                        exclude_user.extend([i] * len(items))
                        exclude_item.extend(items)
                    scores[exclude_user, exclude_item] = -1e5
                    # -----
                    _, neg_item = torch.topk(scores, self.topk)
                    self.RANK[user:user + batch_size] = neg_item
        self.RANK = self.RANK.cpu().int().numpy()

    # ----------------------------------------------------------------------------
    # method 1
    def convex_combine(self, batch_users, batch_pos, batch_neg, epoch):
        with torch.no_grad():
            student_score = userAndMatrix(batch_users, batch_neg, self.student)
            teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)
            start = time()
            batch_list = torch.arange(0, len(batch_neg))
            pos_score = self.teacher(batch_users, batch_pos).unsqueeze(dim=1)
            margin = pos_score - teacher_score
            refine = margin * student_score
            _, student_max = torch.max(refine, dim=1)
            Items = batch_neg[batch_list, student_max]
            return Items, None, None

    # just DNS
    def DNS(self, batch_neg, scores):
        batch_list = torch.arange(0, len(batch_neg))
        _, student_max = torch.max(scores, dim=1)
        student_neg = batch_neg[batch_list, student_max]
        return student_neg

class SimpleSample:
    def __init__(self,
                 dataset: BasicDataset,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns_k: int,
                 method: int = 3,
                 beta=world.beta):

        self.beta = beta

        self.dataset = dataset
        self.student = student
        self.teacher = teacher

        self.dns_k=dns_k

    def PerSample(self, batch=None):
        # if batch is not None:
        #     return UniformSample_DNS_yield(self.dataset,
        #                                    self.dns_k,
        #                                    batch_size=batch)
        # else:
        #     return UniformSample_DNS(self.dataset, self.dns_k)
        return Sample_DNS_python(self.dataset, self.dns_k)

    def Sample(self,
               batch_users,
               batch_pos,
               batch_neg,
               epoch,
               dynamic_samples=None):
        STUDENT = self.student
        TEACHER = self.teacher
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----

        return negitems, None, None



class RD:
    def __init__(
        self,
        dataset: BasicDataset,
        student: PairWiseModel,
        teacher: PairWiseModel,
        dns,
        topK=30,
        mu=0.1,
        lamda=1,
        teach_alpha=1.0,
        dynamic_sample=100,
        dynamic_start_epoch=100,
    ):
        self.rank_aware = False
        self.dataset = dataset
        self.student = student
        self.teacher = teacher.eval()
        self.RANK = None
        self.epoch = 0

        self._weight_renormalize = True
        self.mu, self.topk, self.lamda = mu, topK, lamda
        self.dynamic_sample_num = dynamic_sample
        self.dns_k = dns
        self.teach_alpha = teach_alpha
        self.start_epoch = dynamic_start_epoch
        self._generateTopK()
        self._static_weights = self._generateStaticWeights()

    def PerSample(self, batch=None):
        #S=Sample_DNS_python(self.dataset,self.dynamic_sample_num+1)
        self.dynamic_samples=Sample_neg(self.dataset,self.dynamic_sample_num)
        self.dynamic_samples = torch.Tensor(self.dynamic_samples).long().cuda()
        return Sample_original(self.dataset)

    def _generateStaticWeights(self):
        w = torch.arange(1, self.topk + 1).float()
        w = torch.exp(-w / self.lamda)
        return (w / w.sum()).unsqueeze(0)

    def _generateTopK(self, batch_size=1024):
        if self.RANK is None:
            with torch.no_grad():
                self.RANK = torch.zeros(
                    (self.dataset.n_users, self.topk)).cuda()
                for user in range(0, self.dataset.n_users, batch_size):
                    end = min(user + batch_size, self.dataset.n_users)
                    scores = self.teacher.getUsersRating(
                        torch.arange(user, end))
                    pos_item = self.dataset.getUserPosItems(
                        np.arange(user, end))

                    # -----
                    exclude_user, exclude_item = [], []
                    for i, items in enumerate(pos_item):
                        exclude_user.extend([i] * len(items))
                        exclude_item.extend(items)
                    scores[exclude_user, exclude_item] = -1e5
                    # -----
                    _, neg_item = torch.topk(scores, self.topk)
                    self.RANK[user:user + batch_size] = neg_item

    def _rank_aware_weights(self):
        pass

    def _weights(self, S_score_in_T, epoch, dynamic_scores):
        batch = S_score_in_T.shape[0]
        if epoch < self.start_epoch:
            return self._static_weights.repeat((batch, 1)).cuda()
        with torch.no_grad():
            static_weights = self._static_weights.repeat((batch, 1))
            # ---
            topk = S_score_in_T.shape[-1]
            num_dynamic = dynamic_scores.shape[-1]
            m_items = self.dataset.m_items
            dynamic_weights = torch.zeros(batch, topk)
            for col in range(topk):
                col_prediction = S_score_in_T[:, col].unsqueeze(1)
                num_smaller = torch.sum(col_prediction < dynamic_scores,dim=1).float()
                # print(num_smaller.shape)
                relative_rank = num_smaller / num_dynamic
                appro_rank = torch.floor((m_items - 1) * relative_rank)+1

                dynamic = torch.tanh(self.mu * (appro_rank - col))
                dynamic = torch.clamp(dynamic, min=0.)

                dynamic_weights[:, col] = dynamic.squeeze()
            if self._weight_renormalize:
                return F.normalize(static_weights * dynamic_weights,
                                   p=1,
                                   dim=1).cuda()
            else:
                return (static_weights * dynamic_weights).cuda()

    def Sample(self,
               batch_users,
               batch_pos,
               batch_neg,
               epoch,
               dynamic_samples=None):
        STUDENT = self.student
        TEACHER = self.teacher
        #assert batch_neg.shape[-1] == (self.dns_k)
        dynamic_samples = self.dynamic_samples[batch_users]
        #batch_neg = batch_neg[:, :self.dns_k]
        dynamic_scores = userAndMatrix(batch_users, dynamic_samples,STUDENT).detach()
        topk_teacher = self.RANK[batch_users]
        topk_teacher = topk_teacher.reshape((-1, )).long()
        user_vector = batch_users.repeat((self.topk, 1)).t().reshape((-1, ))

        S_score_in_T = STUDENT(user_vector, topk_teacher).reshape(
            (-1, self.topk))
        weights = self._weights(S_score_in_T.detach(), epoch, dynamic_scores)
        # RD_loss
        RD_loss = -(weights * torch.log(torch.sigmoid(S_score_in_T)))
        # print("RD shape", RD_loss.shape)
        RD_loss = RD_loss.mean(1)
        RD_loss = RD_loss.mean()

        return  RD_loss


class CD:
    def __init__(self,
                 dataset: BasicDataset,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns,
                 lamda=1/20,
                 n_distill=50,
                 t1=2,
                 t2=0):
        self.student = student
        self.teacher = teacher.eval()
        self.dataset = dataset
        self.dns_k = dns
        self.sample_num = 3000
        self.strategy = "student guide"
        self.lamda = lamda
        self.n_distill = n_distill
        self.t1, self.t2 = t1, t2
        #self.sample_num=200
        ranking_list = np.asarray([(i+1)/self.sample_num for i in range(self.sample_num)])
        ranking_list = torch.FloatTensor(ranking_list)
        #self.lamda=1/20
        ranking_list=torch.exp(-(ranking_list*self.lamda))
        self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        self.ranking_mat.requires_grad = False
        if self.strategy == "random":
            MODEL = None
        elif self.strategy == "student guide":
            MODEL = self.student
        elif self.strategy == "teacher guide":
            MODEL = self.teacher
        else:
            raise TypeError("CD support [random, student guide, teacher guide], " \
                            f"But got {self.strategy}")
        with timer(name="CD sample"):
            self.get_rank_sample(MODEL=self.teacher)
    def PerSample(self, batch=None):
        # with timer(name="CD sample"):
        #     self.get_rank_sample(MODEL=self.teacher)
        return  Sample_original(self.dataset)
    def Sample(self, batch_users, batch_pos, batch_neg, epoch):
        return self.sample_diff(batch_users, batch_pos, batch_neg,
                                self.strategy)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items,
                                   (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()

    def get_rank_sample(self,MODEL):
        if MODEL is None:
            return self.random_sample(self.dataset.n_users)
        all_items = self.dataset.m_items
        self.rank_samples = torch.zeros(self.dataset.n_users, self.n_distill)
        with torch.no_grad():
            #items_score = MODEL.getUsersRating(batch_users)
            batch_size=1024
            rank_scores = torch.zeros(self.dataset.n_users, self.sample_num)
            rank_items = torch.zeros(self.dataset.n_users, self.sample_num)
            for user in range(0, self.dataset.n_users, batch_size):
                end = min(user + batch_size, self.dataset.n_users)
                scores = MODEL.getUsersRating(
                    torch.arange(user, end))
                pos_item = self.dataset.getUserPosItems(
                    np.arange(user, end))
                exclude_user, exclude_item = [], []
                for i, items in enumerate(pos_item):
                    exclude_user.extend([i] * len(items))
                    exclude_item.extend(items)
                scores[exclude_user, exclude_item] = -1e10
                rank_scores[user:end],  rank_items[user:end] = torch.topk(scores, self.sample_num)
                del scores

            for user in range(self.dataset.n_users):
                ranking_list = self.ranking_mat[user]
                rating=rank_scores[user]
                negitems=rank_items[user]
                sampled_items = set()
                while True:
                    with timer(name="compare"):
                        samples = torch.multinomial(ranking_list, 2, replacement=True)
                        if rating[samples[0]] > rating[samples[1]]:
                            sampled_items.add(negitems[samples[0]])
                        else:
                            sampled_items.add(negitems[samples[1]])
                        if len(sampled_items)>=self.n_distill:
                            break
                self.rank_samples[user] = torch.Tensor(list(sampled_items))
        self.rank_samples=self.rank_samples.cuda().long()


    def sample_diff(self, batch_users, batch_pos, batch_neg, strategy):
        STUDENT = self.student
        TEACHER = self.teacher

        dns_k = self.dns_k
        random_samples=self.rank_samples[batch_users,:]
        # samples_vector = random_samples.reshape((-1, ))
        samples_scores_T = userAndMatrix(batch_users, random_samples, TEACHER)
        samples_scores_S = userAndMatrix(batch_users, random_samples, STUDENT)
        weights = torch.sigmoid((samples_scores_T + self.t2) / self.t1)
        inner = torch.sigmoid(samples_scores_S)
        CD_loss = -(weights * torch.log(inner + 1e-10) +
                    (1 - weights) * torch.log(1 - inner + 1e-10))
        # print(CD_loss.shape)
        CD_loss = CD_loss.mean(1).mean()
        return CD_loss

    def student_guide(self, batch_users, batch_pos, batch_neg, epoch):
        pass




class RRD:
    def __init__(self,dataset: BasicDataset,
                student: PairWiseModel,
                teacher: PairWiseModel,
                 dns,T=100, K=20, L=20):
        self.student = student
        self.teacher = teacher.eval()

        self.dataset = dataset
        self.dns_k = dns
        # for KD
        self.T = T
        self.K = K
        self.L = L
        ranking_list = np.asarray([(i+1) /30 for i in range(self.T)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list)
        self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        self.ranking_mat.requires_grad = False
        # For uninteresting item
        self.mask = torch.ones((self.dataset.n_users,self.dataset.m_items))
        for user in range(self.dataset.n_users):
            user_pos = self.dataset.getOneUserPosItems(user)
            for item in user_pos:
                self.mask[user][item] = 0
        self.mask.requires_grad = False
        with timer(name="RRD sample"):
            self.RRD_sampling()

    def PerSample(self, batch=None):

        return Sample_original(self.dataset)
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def RRD_sampling(self):
            with torch.no_grad():
                # interesting items
                self.interesting_items = torch.zeros((self.dataset.n_users, self.K))
                for user in range(self.dataset.n_users):
                    crsNeg = np.array(list(self.dataset.getOneUserPosItems(user)))
                    neglist=np.arange(self.dataset.m_items)
                    neglist= np.delete(neglist,crsNeg,-1)
                    negItems = torch.LongTensor(neglist).cuda()
                    n_pos=500
                    samples = torch.multinomial(self.ranking_mat[user,: self.T], self.K, replacement=False)
                    samples = samples.sort(dim=-1)[0].cuda()
                    rating=userAndMatrix(torch.tensor(user), negItems, self.teacher).reshape((-1))
                    n_rat=rating.sort(dim=-1,descending=True)[1]
                    negItems=negItems[n_rat]
                    self.interesting_items[torch.tensor(user)] = negItems[samples]
                    self.mask[user][negItems[: self.T]]=0
                    del samples
                    del negItems
                    del rating
                self.interesting_items = self.interesting_items.cuda()
                # uninteresting items
                m1 = self.mask[:self.dataset.n_users//2,:]
                tmp1 = torch.multinomial(m1, self.L, replacement=False)
                del m1
                m2 = self.mask[self.dataset.n_users//2:,:]
                tmp2 = torch.multinomial(m2, self.L, replacement=False)
                del m2
                self.uninteresting_items =torch.cat((tmp1,tmp2),dim=0).cuda()



    def relaxed_ranking_loss(self,S1,S2):
        above = S1.sum(1, keepdim=True)

        below1 = S1.flip(-1).exp().cumsum(1)
        below2 = S2.exp().sum(1, keepdim=True)

        below = (below1 + below2).log().sum(1, keepdim=True)

        return above - below

    def Sample(self,
               batch_users,
               batch_pos,
               batch_neg, epoch):
        STUDENT = self.student
        TEACHER = self.teacher
        interesting_users, uninteresting_users = self.get_samples(batch_users)

        interesting_user_prediction=torch.sigmoid(userAndMatrix(batch_users, interesting_users.long(), STUDENT))
        uninteresting_user_prediction = torch.sigmoid(userAndMatrix(batch_users, uninteresting_users.long(), STUDENT))
        RRD_loss = self.relaxed_ranking_loss(interesting_user_prediction,uninteresting_user_prediction)
        RRD_loss=-torch.mean(RRD_loss)

        user_de = STUDENT.get_DE_loss(batch_users.long(), is_user=True)
        interesting_users = interesting_users.reshape((-1,)).long()
        interesting_de = STUDENT.get_DE_loss(interesting_users, is_user=False)
        uninteresting_users = uninteresting_users.reshape((-1,)).long()
        uninteresting_de = STUDENT.get_DE_loss(uninteresting_users, is_user=False)
        DE_loss=user_de+interesting_de+uninteresting_de

        return RRD_loss*world.config['kd_weight']+DE_loss*world.config['de_weight']


class HTD:
    def __init__(
        self,
        dataset: BasicDataset,
        student: PairWiseModel,
        teacher: PairWiseModel,
        dns,
        topK=50,
    ):
        self.rank_aware = False
        self.dataset = dataset
        self.student = student
        self.teacher = teacher.eval()
        self.RANK = None
        self.epoch = 0


    def PerSample(self, batch=None):
        #S=Sample_DNS_python(self.dataset,self.dynamic_sample_num+1)
        return Sample_original(self.dataset)

    def Sample(self,
               batch_users,
               batch_pos,
               batch_neg,
               epoch):

        GA_loss_user = self.student.get_GA_loss(batch_users.unique(), is_user=True)
        GA_loss_item = self.student.get_GA_loss(torch.cat([batch_pos, batch_neg], 0).unique(), is_user=False)
        GA_loss = GA_loss_user + GA_loss_item

        ## Topology Distillation
        TD_loss = self.student.get_TD_loss(batch_users.unique(), torch.cat([batch_pos, batch_neg], 0).unique())
        HTD_loss = TD_loss * world.config['alpha'] + GA_loss * (1 - world.config['alpha'])

        return  HTD_loss




class UnKD:
    def __init__(self,
                 dataset: Loader,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns,
                 lamda=1.0,
                 n_distill=30):
        self.student = student
        self.teacher = teacher.eval()
        self.dataset = dataset
        self.dns_k = dns
        self.lamda = world.config['lamda']
        self.n_distill = world.config['sample_num']
        with timer(name="DD sample"):
            self.get_rank_negItems()


    def PerSample(self, epoch=1):
        self.KD_sampling()
        return Sample_original(self.dataset)

    def Sample(self, batch_users, batch_pos, batch_neg, epoch):
        return self.sample_diff(batch_users, batch_pos, batch_neg)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items,
                                   (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()

    def get_rank_negItems(self):
        self.group_ratio=self.dataset.group_ratio
        all_ratio=0.0
        for i in range(len(self.group_ratio)):
            self.group_ratio[i]=math.pow(self.group_ratio[i],self.lamda)
            all_ratio+=self.group_ratio[i]
        for i in range(len(self.group_ratio)):
            self.group_ratio[i]=self.group_ratio[i]/all_ratio
        print(self.group_ratio)
        all_n =0
        for i in self.group_ratio:
            all_n += round(i * self.n_distill)
        print(all_n)
        if all_n<self.n_distill:
            all_n=self.n_distill
        ranking_list = np.asarray([(1 + i) / 20 for i in range(1000)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list)
        #self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        #ranking_list.requires_grad = False
        self.ranking_list=ranking_list
        self.ranking_list.requires_grad = False
        self.user_negitems = [list() for u in range(self.dataset.n_users)]

        self.pos_items = torch.zeros((self.dataset.n_users, all_n))
        self.neg_items = torch.zeros((self.dataset.n_users, all_n))
        self.item_tag = torch.zeros((self.dataset.n_users, all_n))
        for i in range(len(self.dataset.item_group)):
            cate_items = set(self.dataset.item_group[i])
            ratio = self.group_ratio[i]
            dis_n = math.ceil(self.n_distill * ratio)
            for user in range(self.dataset.n_users):
                crsNeg = set(list(self.dataset.getOneUserPosItems(user)))
                neglist = list(cate_items - crsNeg)
                negItems = torch.LongTensor(neglist)
                rating = userAndMatrix(torch.tensor(user), negItems, self.teacher).reshape((-1))
                n_rat = rating.sort(dim=-1, descending=True)[1]
                negItems = negItems[n_rat]
                self.user_negitems[user].append(negItems[:1000])

    def KD_sampling(self):
        with torch.no_grad():
            # interesting items

            pos_items = [list() for u in range(self.dataset.n_users)]
            neg_items = [list() for u in range(self.dataset.n_users)]
            item_tag = [list() for u in range(self.dataset.n_users)]
            all_n = 0
            for i in self.group_ratio:
                all_n += round(i * self.n_distill)

            for i in range(len(self.dataset.item_group)):
                ratio=self.group_ratio[i]
                dis_n=round(ratio*self.n_distill)
                if all_n<self.n_distill:
                    tm=self.n_distill-all_n
                    if i<tm:
                        dis_n+=1
                for user in range(self.dataset.n_users):
                    temp_ranklist=self.ranking_list.clone()
                    user_weight=self.dataset.user_group_ratio[user][i]
                    negItems =self.user_negitems[user][i]

                    with timer(name="compare"):
                        while True:
                            k = 0
                            samples1 = torch.multinomial(temp_ranklist[:len(negItems)], dis_n, replacement=True)
                            #elec
                            #temp_ranklist[:10]=0
                            samples2 = torch.multinomial(temp_ranklist[:len(negItems)], dis_n, replacement=True)
                            for l in range(len(samples1)):
                                if samples1[l] < samples2[l]:
                                    k+=1
                                elif samples1[l] > samples2[l]:
                                    k+=1
                                    s=samples1[l]
                                    samples1[l]=samples2[l]
                                    samples2[l]=s
                            if k>=dis_n:
                                pos_items[user].extend(negItems[samples1])
                                neg_items[user].extend(negItems[samples2])
                                item_tag[user].extend([user_weight]*len(samples1))
                                break
            for user in range(self.dataset.n_users):
                self.pos_items[user] = torch.Tensor(pos_items[user])
                self.neg_items[user] =torch.Tensor(neg_items[user])
                self.item_tag[user] =torch.Tensor(item_tag[user])
        self.pos_items = self.pos_items.long().cuda()
        self.neg_items =self.neg_items.long().cuda()
        self.item_tag = self.item_tag.cuda()

    def sample_diff(self, batch_users, batch_pos, batch_neg):
        STUDENT = self.student
        TEACHER = self.teacher

        pos_samples=self.pos_items[batch_users]
        neg_samples = self.neg_items[batch_users]
        weight_samples = self.item_tag[batch_users]
        pos_scores_S = userAndMatrix(batch_users, pos_samples, STUDENT)
        neg_scores_S = userAndMatrix(batch_users, neg_samples, STUDENT)
        mf_loss = torch.log(torch.sigmoid(pos_scores_S - neg_scores_S) + 1e-10)
        mf_loss = torch.mean(torch.neg(mf_loss),dim=-1)
        #mf_loss = torch.mean(torch.neg(mf_loss), dim=-1)
        mf_loss=torch.mean(mf_loss)
        return mf_loss


# ==============================================================
# NON-EXPERIMENTAL PART
# ==============================================================


# ----------
 # sample
def Sample_original(dataset):
    """
    the original implement of BPR Sampling in LightGCN
    NOTE: we sample a whole epoch data at one time
    :return:
        np.array
    """
    return Sample_DNS_python(dataset,1)

# ----------

def Sample_DNS_python(dataset,dns_k):
    """python implementation for 'UniformSample_DNS'
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    allPos = dataset.allPos
    S = []
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for user in range(dataset.n_users):
        posForUser = list(allPos[user])
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        per_user_num=len(posForUser)
        for i in range(per_user_num):
            positem = posForUser[i]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            add_pair = [user, positem, *negitems]
            S.append(add_pair)
    return S


def Sample_DNS_cate_epoch(dataset,dns_k,epoch):
    """python implementation for 'UniformSample_DNS'
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    allPos = dataset.allPos
    itemcate = dataset.getOneCate(torch.arange(dataset.m_items))
    S = []
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for user in range(dataset.n_users):
        posForUser = list(allPos[user])
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        per_user_num=len(posForUser)
        for i in range(per_user_num):
            positem = posForUser[i]
            poscatelen=len(itemcate[positem])
            poscate=itemcate[positem][epoch%poscatelen]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            negcatelen = len(itemcate[int(negitems)])
            negcate=itemcate[int(negitems)][epoch%negcatelen]
            #negcate = np.random.choice(itemcate[int(negitems)], size=1, replace=False)
            add_pair = [user, positem, int(negitems),poscate,negcate]
            S.append(add_pair)
    return S







def Sample_neg(dataset,dns_k):
    """python implementation for 'UniformSample_DNS'
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    allPos = dataset.allPos
    S = []
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for user in range(dataset.n_users):
        posForUser = list(allPos[user])
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
        negitems = NEGforUser[negindex]
        add_pair = [*negitems]
        S.append(add_pair)
    return S

def Sample_neg_cate(dataset):
    """python implementation for 'UniformSample_DNS'
    """
    dataset: BasicDataset
    item_num = dataset.m_items
    allPos = dataset.getOneCate(torch.arange(dataset.m_items))
    S = []
    BinForUser = np.zeros(shape=(len(dataset.getAllCate()), )).astype("int")
    for item in range(dataset.m_items):
        posForUser = allPos[item]
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        BinForUser[0]=1
        NEGforUser = np.where(BinForUser == 0)[0]
        # for j in range(len(posForUser)):
        #     poscate=posForUser[j]
        #     negindex = np.random.randint(0, len(NEGforUser), size=(1, ))
        #     negicate = NEGforUser[negindex]
        #     add_pair = [item,poscate,negicate]
        #     S.append(add_pair)
        negindex = np.random.randint(0, len(NEGforUser), size=(1,))
        negicate = NEGforUser[negindex]
        add_pair = [item,negicate]
        S.append(add_pair)
    return S

def DNS_sampling_neg(batch_users, batch_neg, dataset, recmodel):
    """Dynamic Negative Choosing.(return max in Neg)

    Args:
        batch_users ([tensor]): shape (batch_size, )
        batch_neg ([tensor]): shape (batch_size, dns_k)
        dataset ([BasicDataset])
        recmodel ([PairWiseModel])

    Returns:
        [tensor]: Vector of negitems, shape (batch_size, ) 
                  corresponding to batch_users
    """
    dns_k = world.DNS_K
    with torch.no_grad():

        scores = userAndMatrix(batch_users, batch_neg, recmodel)

        _, top1 = scores.max(dim=1)
        idx = torch.arange(len(batch_users)).cuda()
        negitems = batch_neg[idx, top1]
    return negitems


