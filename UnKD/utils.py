'''
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import os

#from fairsearchcore.models import FairScoreDoc

import world
import torch
import random
import numpy as np
from tqdm import tqdm
from time import time
from torch import nn, optim
from torch import log, Tensor

from dataloader import BasicDataset, Loader


# ============================================================================
# ============================================================================
# pair loss
class BPRLoss:
    def __init__(self, recmodel, config: dict,optloss):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.loss_weight = 1
        self.add_weight = config['kd_weight']
        self.self_weight =1-config['kd_weight']
        self.lr = config['lr']
        self.opt=optloss


    def stageOne(self,
                 users,
                 pos,
                 neg,epoch,add_loss=None):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        if add_loss is not None:
            assert add_loss.requires_grad == True
            if world.SAMPLE_METHOD == 'DE_RRD':
                loss=loss+add_loss
            elif world.SAMPLE_METHOD == 'RD':
                loss = loss*self.self_weight + add_loss * self.add_weight
            else:
                loss = loss + add_loss* self.add_weight
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

    def stageTwo(self,
                 users,
                 pos,
                 neg, epoch, add_loss=None):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        if add_loss is not None:
            assert add_loss.requires_grad == True
            loss = loss + add_loss * self.add_weight
        loss.backward()
        self.opt.step()
        return loss.cpu().item()


def getTestweight(users: Tensor, items: Tensor, dataset: BasicDataset):
    """
        designed only for levave-one-out data
    """
    users = users.cpu().numpy().astype('int')
    items = items.cpu().numpy().astype('int')
    testdict = dataset.testDict
    test_items = []
    for user in users:
        test_item = testdict[user][0]
        test_items.append(test_item)
    test_items = np.array(test_items).astype('int')
    index = (test_items == items)
    weights = np.ones_like(users)
    weights[index] = world.ARGS.testweight

    return Tensor(weights).cuda()


# ============================================================================
# ============================================================================
# utils
def mat_to_list(base_mat):
    result = []

    for user_id in range(base_mat.shape[0]):
        for item_id in range(base_mat.shape[1]):
            if base_mat[user_id][item_id] == 1:
                result.append((user_id, item_id, 1))
    return result

def is_visited(base_dict, user_id, item_id):
	if user_id in base_dict and item_id in base_dict[user_id]:
		return True
	else:
		return False

class EarlyStop:
    def __init__(self, patience, model, filename):
        self.patience = patience
        self.model = model
        self.best_model = model.state_dict()
        self.filename = filename
        self.suffer = 0
        self.best = 0
        self.best_result = None
        self.best_epoch = 0
        self.mean = 0
        self.sofar = 1
        print('stopflag:'+world.stopflag)
        self.best_other=None
        self.meanLoss=10
        self.bestLoss=0

    def step(self, epoch, performance,other):
        if performance['ndcg'][-1] < self.mean:
            self.suffer += 1
            if self.suffer >= self.patience:
                return True
        else:
            self.suffer = 0
            self.mean = performance['ndcg'][-1]
            self.sofar = 1
            self.best = performance['ndcg'][-1]
            self.best_result = performance
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()
            self.best_other=other
            torch.save(self.best_model, self.filename)
            return False

    def stepLoss(self, epoch, loss):
        if loss > self.meanLoss:
            self.suffer += 1
            if self.suffer >= self.patience:
                return True
        else:
            self.suffer = 0
            self.meanLoss = loss
            self.bestLoss = loss
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()
            torch.save(self.best_model, self.filename)
            return False

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName(model_name, dataset, rec_dim, layers=None, dns_k=None):
    if model_name == 'mf':
        if dns_k is not None:
            file = f"mf-{dataset}-{rec_dim}-{dns_k}"
        else:
            file = f"mf-{dataset}-{rec_dim}"
    elif model_name == 'lgn':
        assert layers is not None
        if dns_k is not None:
            file = f"lgn-{dataset}-{layers}-{rec_dim}-{dns_k}"
        else:
            file = f"lgn-{dataset}-{layers}-{rec_dim}"
    else:
        assert layers is not None
        if dns_k is not None:
            file = f"{model_name}-{dataset}-{layers}-{rec_dim}-{dns_k}"
        else:
            file = f"{model_name}-{dataset}-{layers}-{rec_dim}"
    file = file + f".pth.tar"
    return file


def getLogFile():
    model = world.model_name
    comment = world.comment
    dataset = world.datasetpkl
    return f"{dataset}-{model}-{comment}.txt"


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have ' 'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def TO(*tensors, **kwargs):
    results = []
    for tensor in tensors:
        results.append(tensor.cuda())
    return results


def shapes(*tensors):
    shape = [tensor.size() for tensor in tensors]
    strs = [str(sh) for sh in shape]
    print(" : ".join(strs))


def getTeacherConfig(config: dict):
    teacher_dict = config.copy()
    teacher_dict['lightGCN_n_layers'] = teacher_dict['teacher_layer']
    teacher_dict['latent_dim_rec'] = teacher_dict['teacher_dim']
    return teacher_dict


def time2str(sam_time: list):
    sam_copy = ""
    for t in sam_time:
        sam_copy += '+' + f"{t:.2f}"
    return sam_copy[1:]


# Draw and Count<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def draw_longtail(dataset, pop1, pop2):
    import matplotlib.pyplot as plt
    dataset: Loader
    # pop_item, index = dataset.popularity()
    x = dataset.popularity()[0]
    x = x / x.max()
    pop1 = pop1 / pop1.max()
    pop2 = pop2 / pop2.max()
    print(x.sum(), pop1.sum(), pop2.sum())

    pop1 = np.sort(pop1)[::-1]
    pop2 = np.sort(pop2)[::-1]
    x = np.sort(x)[::-1]

    plt.plot(range(len(x)), pop1, label="student")
    plt.plot(range(len(x)), pop2, label="RD")
    plt.plot(range(len(x)), x, label="dataset")
    plt.legend()
    plt.show()


def draw(dataset, pop1, pop2, name1='pop1', name2='pop2'):
    import matplotlib.pyplot as plt
    import powerlaw
    dataset: Loader
    # pop_item, index = dataset.popularity()
    x = dataset.popularity()[0]
    print(x.sum(), pop1.sum(), pop2.sum())

    # x = x/x.max()
    # norm = max(pop1.max(), pop2.max())
    # pop1 = pop1 / pop1.max()
    # pop2 = pop2 / pop2.max()

    pop1_mask = (pop1 > x)
    pop2_mask = (pop2 > x)

    plt.scatter(x[~pop1_mask],
                pop1[~pop1_mask],
                c='springgreen',
                linewidth=0,
                s=10,
                alpha=1,
                label=name1)
    plt.scatter(x[~pop2_mask],
                pop2[~pop2_mask],
                c='blue',
                s=10,
                linewidth=0,
                alpha=0.3,
                label=name2)

    plt.scatter(x[pop2_mask],
                pop2[pop2_mask],
                c='blue',
                s=30,
                linewidth=0,
                alpha=0.8,
                label=name2)
    plt.scatter(x[pop1_mask],
                pop1[pop1_mask],
                c='springgreen',
                linewidth=0,
                s=30,
                alpha=1,
                label=name1)

    plt.plot(x, x, linewidth=2, label="dataset")
    plt.xlabel("Dataset popularity rate")
    plt.ylabel("Model popularity rate")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 4, 2, 3]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    plt.legend(handles, labels)
    plt.title("Gowalla")
    plt.show()


def powerlaw(pop1, pop2, pop3):
    import matplotlib.pyplot as plt
    import powerlaw
    fit1 = powerlaw.Fit(pop1)
    fit2 = powerlaw.Fit(pop2)
    fit3 = powerlaw.Fit(pop3)
    fig1 = fit1.plot_pdf(color='b', label="Teacher")
    fit1.power_law.plot_pdf(color='b', linestyle="--", ax=fig1)

    # fig1 = fit1.plot_ccdf(color='b', label="Teacher")
    # fit1.power_law.plot_ccdf(color='b', linestyle="--", ax=fig1)

    fit2.plot_pdf(color='y', ax=fig1, label="RD-32")
    fit2.power_law.plot_pdf(color='y', linestyle="--", ax=fig1)
    fit3.plot_pdf(color='r', ax=fig1, label="Student")
    fit3.power_law.plot_pdf(color='r', linestyle="--", ax=fig1)

    # fit2.plot_ccdf(color='y', ax=fig1, label="RD-32")
    # fit2.power_law.plot_ccdf(color='y', linestyle="--", ax=fig1)
    # fit3.plot_ccdf(color='r', ax=fig1, label="Student")
    # fit3.power_law.plot_ccdf(color='r', linestyle="--", ax=fig1)

    plt.xlabel("log(x)")
    plt.ylabel("log(#popularity)")
    plt.title("Probability Density Function")
    # plt.title("Complementary Cumulative Distribution Function")
    plt.legend()
    plt.show()


def map_item_three(pop_item):
    """mapping item into short-head(0.2), long-tail(0.6), distant-tail(0.2)

    Args:
        pop_item ([type]): [description]
        
    Return:
        list[ndarray...]: short-head, long-tail, distant-tail
    """
    return map_item_N(pop_item, [0.2, 0.6, 0.2])


def map_item_N(pop_item, spilt):
    """mapping item into 

    Args:
        pop_item ([type]): [description]
        
    Return:
        list[ndarray...]: short-head, long-tail, distant-tail
    """
    from math import floor, ceil

    #assert sum(spilt) == 1.
    index = np.argsort(pop_item)[::-1]
    num_item = len(index)
    # return (set(index[:floor(num_item * 0.2)]),
    #         set(index[ceil(num_item * 0.2):floor(num_item * 0.8)]),
    #         set(index[ceil(num_item * 0.8):]))
    mapping = []

    for i in range(len(spilt)):
        left = floor(sum(spilt[:i]) * num_item)
        right = floor(left + spilt[i] * num_item)
        mapping.append(set(index[left:right]))
    return mapping


def APT(pop_user, mappings):
    """calculate the APT metrics for different sets

    Args:
        pop_user (list | ndarray): the recommend list or history of users
        mappings (list | tuple): (short-head, long-tail, distant-tail)

    Returns:
        list: APTs for different mappings
    """
    total_set = len(mappings)
    total_user = len(pop_user)
    apts = []
    for mapping in mappings:
        apt = 0.
        for user_item in pop_user:
            count = list(map(lambda x: x in mapping, user_item))
            apt += np.mean(count)
        apt = apt / total_user
        apts.append(apt)
    return apts


def popularity_ratio(pop_model: np.ndarray, pop_model_user: np.ndarray,
                     dataset: Loader):
    """calculate the degree of the "long-tailness" for a distribution

    Args:
        pop_model (ndarray): the freq of items recommended by model
        pop_model_user (ndarray): (user X topk) the recommend list of users
        dataset (dataloader.Loader): the freq of items in dataset

    Returns:
        dict: {"I_ratio""float, "I_KL":float, "I_gini":float, "APT":[], "I_bin": float}
    """

    pop_dataset= dataset.itemCount()
    assert len(pop_model) == len(pop_dataset)
    num_item = len(pop_dataset)
    num_interaction = pop_model.sum()
    metrics = {}

    prop_model = pop_model / num_interaction
    prop_uniform = 1. / num_item

    prop_dataset = pop_dataset / pop_dataset.sum()
    # print("dataset KL",
    #       np.sum(prop_dataset * np.log(prop_dataset / prop_uniform + 1e-7)))
    metrics['I_KL'] = np.sum(prop_model *
                             np.log(prop_model / prop_uniform + 1e-7))

    mapping = map_item_three(pop_dataset)
    mapping_N = map_item_N((pop_dataset),
                           [0.2, 0.2, 0.2, 0.2, 0.2])

    metrics['APT'] = APT(pop_model_user, mapping)
    metrics['APT5'] = APT(pop_model_user, mapping_N)

    # print("dataset APT", APT(dataset.allPos, mapping))
    # print("dataset bin", np.sum(pop_dataset/pop_dataset.max()))
    metrics['I_bin'] = np.sum(pop_model / np.max(pop_model))

    return metrics


# Dataset spliting (only used once for generation)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def _loo_split_dataset(train_f, test_f):
    train = {}
    test = {}
    with open(train_f, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split()
                item = [int(i) for i in line[1:]]
                user = int(line[0])
                train[user] = item
    with open(test_f, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split()
                item = [int(i) for i in line[1:]]
                user = int(line[0])
                test[user] = item
    for key in list(test):
        train[key] = train[key] + test[key]

    all_data = train
    train_seq = []
    valid_seq = []
    test_seq = []
    users_list = sorted(list(train))
    for user in users_list:
        user_item = all_data[user]
        assert len(user_item)
        for t_item in user_item[:-2]:
            train_seq.append((user, t_item, 1))
        valid_seq.append((user, user_item[-2], 1))
        test_seq.append((user, user_item[-1], 1))
    train_file = "bptrain.txt"
    valid_file = "bpvalid.txt"
    test_file = "bytest.txt"
    with open(train_file, 'w') as f:
        for t_data in train_seq:
            f.write(f"{t_data[0]} {t_data[1]} {1}\n")
    with open(valid_file, 'w') as f:
        for t_data in valid_seq:
            f.write(f"{t_data[0]} {t_data[1]} {1}\n")
    with open(test_file, 'w') as f:
        for t_data in test_seq:
            f.write(f"{t_data[0]} {t_data[1]} {1}\n")


def _split_dataset(train_f, ratio=.1):
    train = {}
    with open(train_f, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split()
                item = [int(i) for i in line[1:]]
                user = int(line[0])
                train[user] = item
    valid = {}
    for user in list(train):
        items = np.array(train[user]).astype('int')
        valid_num = round(ratio * len(items))
        if valid_num > 0:
            index = np.random.choice(np.arange(len(items)),
                                     size=(valid_num, ),
                                     replace=False)
            train_index = np.ones_like(items).astype('bool')
            train_index[index] = False
            train_items = items[train_index]
            valid_items = items[~train_index]
            train[user] = train_items
            valid[user] = valid_items
        else:
            print(f"{user} has no items in VALID")

    train_file = "bptrain.txt"
    valid_file = "bpvalid.txt"
    with open(train_file, 'w') as f:
        users = sorted(list(train))
        for user in users:
            f.write(f"{user}")
            for u_item in train[user]:
                f.write(f" {int(u_item)}")
            f.write('\n')
    with open(valid_file, 'w') as f:
        users = sorted(list(valid))
        for user in users:
            f.write(f"{user}")
            for u_item in valid[user]:
                f.write(f" {int(u_item)}")
            f.write('\n')


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


def load(model, file):
    try:
        model.load_state_dict(torch.load(file))
    except RuntimeError:
        model.load_state_dict(
            torch.load(file, map_location=torch.device('cpu')))
    except FileNotFoundError:
        raise FileNotFoundError(f"{file} NOT exist!!!")


def combinations(start, end, com_num=2):
    """get all the combinations of [start, end]"""
    from itertools import combinations
    index = np.arange(start, end)
    return np.asanyarray(list(combinations(index, com_num)))


def display_top(snapshot, key_type='lineno', limit=3):
    import tracemalloc
    import linecache

    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB" %
              (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# ============================================================================
# ============================================================================
# metrics
def getAllDivmetrics(dataset,rankdict,truedict,topks):
    results={}
    rc, cc,ild,FMeasure = [], [],[],[]

    for k in topks:
        actualDict = {}
        for j in rankdict.keys():
            actualDict[j]=rankdict[j][:k]
        #aNDCG.append(getAlphaNDCG(dataset, actualDict,k))
        cc.append(getCC(dataset, actualDict,k))
        ild.append(getILD(dataset, rankdict,k))
        rc.append(getRC(dataset,rankdict,truedict,k))
        #FMeasure.append(getFMeasure(dataset, rankdict, truedict,k))
    #results['AlphaNDCG'] = aNDCG
    #results['FMeasure'] = FMeasure
    results['CC'] = cc
    results['ILD'] = ild
    results['RC'] = rc
    return  results

def getAlphaNDCG(dataset,rankingDict,topk): #类似于预测category的ndcg
    result=0
    queryDict = dataset.all_cate_list
    myAlpha = AlphaNDCG(query_topics=queryDict, doc_topics=dataset.item_cate_dict_s, alpha=0.5)
    myAlpha.calculate_Alpha_nDCG(ranking_query_doc=rankingDict, depth=topk)
    for query in myAlpha.ndcg_values:
         result+=myAlpha.ndcg_values[query][-1]
    return result/(len(rankingDict))

def getRC(dataset,rankdict, gtItems,topk): #类似于预测category的recall
    result=0
    for i in rankdict.keys():
        #item_cate_arr = np.array(list(dataset.item_cate_dict_s.values())).reshape(-1)
        item_cate_arr =list(dataset.item_cate_dict_s.values())
        rank_catelist=[]
        gt_catelist=[]
        for j in range(len(rankdict[i][:topk])):
            rank_catelist=rank_catelist+item_cate_arr[rankdict[i][j]]
        for k in range(len(gtItems[i])):
            gt_catelist=gt_catelist+item_cate_arr[gtItems[i][k]]
        rank_catelist=set(rank_catelist)
        gt_catelist = set(gt_catelist)
        result += len(rank_catelist & gt_catelist) / len(gt_catelist)
    result=result/len(rankdict)
    return result

def getCC(dataset,rankdict,topk): #类别覆盖率
    result = 0
    item_cate_dict = dataset.item_cate_dict_s
    cate_list = dataset.all_cate_list
    item_cate_arr = list(item_cate_dict.values())
    for i in rankdict.keys():
        rank_catelist = []
        for j in range(len(rankdict[i])):
            rank_catelist = rank_catelist + random.sample(item_cate_arr[rankdict[i][j]],1)
        rank_catelist = list(set(rank_catelist))
        curent_r = 0
        for j in range(len(rank_catelist)):
            if rank_catelist[j] in cate_list:
                curent_r += 1
        curent_r = curent_r / (len(cate_list))
        result += curent_r
    result = result / len(rankdict)
    return result



def getILD(dataset,rankdict,topk):
    result=0
    item_cate_dict = dataset.item_cate_dict_s
    cate_list = dataset.all_cate_list
    item_cate_arr = list(item_cate_dict.values())
    for i in rankdict.keys():
        K = len(rankdict[i][:topk])
        tmp = 0
        for j in range(len(rankdict[i][:topk-1])):
            cate_j=set(item_cate_arr[rankdict[i][j]])
            for k in range(j+1, len(rankdict[i][:topk])):
                cate_k=set(item_cate_arr[rankdict[i][k]])
                tmp+=1-(len((cate_j & cate_k))/len((cate_j | cate_k))) #两个item的category vec计算cos距离，相当于相同category的加上1
        if K==1:
            tmp=1
        else:
            tmp = 2*tmp/(K*(K-1))
        result+=tmp
    result=result/len(rankdict)
    return result

def getNTC(dataset,rankdict,topk): #类别覆盖率标准化
    result=0
    maxK=-1
    minK=1e4
    item_cate_dict=dataset.item_cate_dict
    cate_list=dataset.all_cate_list
    for i in rankdict.keys():
        item_cate_arr = np.array(list(item_cate_dict.values())).reshape(-1)
        rank_catelist = list(set(item_cate_arr[rankdict[i][:topk]]))
        curent_r= 0
        for j in range(len(rank_catelist)):
            if rank_catelist[j] in cate_list:
                curent_r+=1
        curent_r=curent_r/(len(cate_list))
        if curent_r>maxK:
            maxK=curent_r
        if curent_r<minK:
            minK=curent_r
        result += curent_r
    curent_r=(result-minK*len(rankdict))/(maxK-minK)
    result=curent_r/len(rankdict)
    return result

def getFMeasure(dataset,rankdict,truedict,topk):
    result=0
    ranklist=[i for i in rankdict.values()]
    truelist=[i for i in truedict.values()]
    queryDict = dataset.all_cate_list
    myAlpha = AlphaNDCG(query_topics=queryDict, doc_topics=dataset.item_cate_dict_s, alpha=0.5)
    myAlpha.calculate_Alpha_nDCG(ranking_query_doc=rankdict, depth=topk)
    l = getLabel(truelist, ranklist)
    accuracy=getNDCG(ranklist, l, topk)
    diversity=np.array([i for i in myAlpha.ndcg_values.values()])
    diversity=diversity[:,-1]
    result+=(2*accuracy*diversity)/(accuracy+diversity)
    return np.sum(result)/(len(rankdict))

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def RecallAndPrecisionByGroup(dataset,test_data,pre_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    #result=[0.0 for j in range(len(dataset.cate_group))]
    recallResult = [0.0 for j in range(len(dataset.item_group))]
    precisionResult = [0.0 for j in range(len(dataset.item_group))]
    for i in range(len(test_data)):
        tempcatelen = [0 for j in range(len(dataset.item_group))]
        precatelen = [0 for j in range(len(dataset.item_group))]
        for j in range(len(dataset.item_group)):
            s1=len(set(test_data[i])&set(dataset.item_group[j]))
            tempcatelen[j]=s1
            s2 = len(set(pre_data[i,:k])&set(test_data[i]) & set(dataset.item_group[j]))
            precatelen[j]=s2
            if tempcatelen[j]!=0:
                recallResult[j]+=precatelen[j]/tempcatelen[j]
            precisionResult[j] += precatelen[j] / k
    return np.array(recallResult),np.array(precisionResult)


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def getNDCG(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return ndcg

def NDCGatK_r_ONE(r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    pred_data = r[:, :k]

    idcg = 1
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def HRatK_ONE(r, k):
    pred = r[:, :k]
    return np.sum(pred)


def getLabel_ONE(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i][0]
        predictTopK = pred_data[i]
        pred = (predictTopK == groundTrue)
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def getSimilaritiyMatrix(dataset):
    m=np.zeros((dataset.m_items, dataset.m_items))
    item_cate_dict = dataset.item_cate_dict_s
    item_cate_arr = list(item_cate_dict.values())
    for i in range(dataset.m_items-1):
        cate_i= set(item_cate_arr[i])
        for j in range(i+1,dataset.m_items):
            cate_j = set(item_cate_arr[j])
            m[i,j]= (len((cate_i & cate_j)) / len((cate_i | cate_j)))*0.1
            m[j,i] =(len((cate_i & cate_j)) / len((cate_i | cate_j)))*0.1
    for i in range(dataset.m_items):
        m[i,i]=1
    return m



def split_item_popularity(pop_item,n):
    """mapping item into

    Args:
        pop_item ([type]): [description]

    Return:
        list[ndarray...]: short-head, long-tail, distant-tail
    """
    from math import floor, ceil
    index =np.argsort(pop_item)[::-1]
    sum_pop=np.sum(pop_item)
    ave_pop=sum_pop/n
    mapping = []
    k=0
    left=0
    i=0
    for i in range(len(index)):
        k=k+pop_item[index[i]]
        if k>ave_pop:
            mapping.append(set(index[left:i-1]))
            left=i
            k=pop_item[index[i]]
    return mapping


def PopularityByGrpup(pre_data,dataset: Loader):
    popularity = dataset.itemCount()
    popularity=popularity.astype(np.float)
    popularity+=1.0
    sum_popularit = np.sum(popularity)
    popularity = popularity / sum_popularit
    popularity = np.power(popularity, 1)
    #popularity = np.power(popularity, 1)
    mappings=split_item_popularity(popularity,5)
    metrics = {}
    total_user = len(pre_data)
    max_K = max(world.topks)
    apts=[]
    for mapping in mappings:
        apt = 0.
        for i in range(len(pre_data)):
            groundnum = pre_data[i]
            #groundnum = range(0,10)
            if len(groundnum)>0:
                count = list(map(lambda x: x in mapping, groundnum))
                apt += np.sum(count)/max_K
        apt = apt / total_user
        apts.append(apt)
    #metrics['precisionbygroup']=APT(groundTrue,mappings=mapping)
    return apts
# ====================end Metrics=============================
def MMR(rel_score,smi_score,lamda=0.5,topk=30):
    rel_score=rel_score.t()

    result=[]
    _,index=torch.topk(rel_score,1,dim=0)
    result.append(index)
    rel_score[index]=0

    while True:
        MMRScore = torch.zeros((len(rel_score)))
        for i in range(len(rel_score)):
            if rel_score[i]<1e-4:
                continue
            maxs = torch.tensor([0]).cuda()
            for j in range(len(result)):
                s=smi_score[i,result[j]]
                maxs=max(maxs,s)
            MMRScore[i]=lamda*rel_score[i]-(1-lamda)*maxs
        _, index = torch.topk(MMRScore, 1,dim=0)
        result.append(index)
        rel_score[index] = 0
        if len(result)>=topk:
            break
    return result

# =========================================================
#import fairsearchcore as fsc

def getFairRank(fair,rankdict,ratingdict,dataset,topk):
    resultDict = {}

    for i in rankdict.keys():
        # item_cate_arr = np.array(list(dataset.item_cate_dict_s.values())).reshape(-1)
        unfair_ranking=[]
        for j in range(len(rankdict[i])):
            if rankdict[i][j] in dataset.item_group[0]:
                t=False
            else:
                t=True
            fairdoc=FairScoreDoc(rankdict[i][j],ratingdict[i][j],t)
            unfair_ranking.append(fairdoc)
        re_ranked = fair.re_rank(unfair_ranking)
        resultDict[i]=[]
        for k in re_ranked:
            resultDict[i].append(k.id)
    return resultDict


def getOneNDCG(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    pred_data = r[:k]

    test_matrix = np.zeros((1, k))
    length = k if k <= len(test_data) else len(test_data)
    test_matrix[:length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return ndcg

def getOneDCG(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    pred_data = r[:k]

    test_matrix = np.zeros((1, k))
    length = k if k <= len(test_data) else len(test_data)
    test_matrix[:length] = 1
    max_r = test_matrix
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg)
    return dcg


def resultBygroup(rankDict,dataset,k):
    result=[0.0 for i in range(len(dataset.item_group))]
    for i in rankDict.keys():
        for j in range(len(dataset.item_group)):
            result[j]+=len(set(rankDict[i][:k])&set(dataset.item_group[j]))
    tot=sum(result)
    for i in range(len(result)):
        result[i]=result[i]/tot
    return result



def getDTR(rankDict,ratingDict,trueDict,dataset,topk):
    result=0
    group_1_sum=0
    group_2_sum = 0
    dcg1_sum=0
    dcg2_sum = 0
    ex1_sum=0
    ex2_sum=0
    for i in rankDict.keys():
        group_1_testdata=list(set(trueDict[i])&set(dataset.popular_group))
        group_2_testdata = list(set(trueDict[i])-set(group_1_testdata))

        ex1=list(map(lambda x: x in dataset.popular_group, rankDict[i][:topk]))
        l=[1 for j in range(topk)]
        ex2 = list(map(lambda x: x[0]-x[1], zip(l, ex1)))
        ex1 = np.array(ex1).astype("float")
        ex2 = np.array(ex2).astype("float")
        ex_dcg_1 = ex1 * (1. / np.log2(np.arange(2, topk + 2)))
        ex_dcg_1 = np.sum(ex_dcg_1)
        ex_dcg_2 = ex2 * (1. / np.log2(np.arange(2, topk + 2)))
        ex_dcg_2 = np.sum(ex_dcg_2)
        ex_n_1=sum(ex1)
        ex_n_2=sum(ex2)

        r1=list(map(lambda x: x in group_1_testdata, rankDict[i][:topk]))
        r2 = list(map(lambda x: x in group_2_testdata, rankDict[i][:topk]))
        r1 = np.array(r1).astype("float")
        r2 = np.array(r2).astype("float")
        pr_dcg_1 = r1 * (1. / np.log2(np.arange(2, topk + 2)))
        pr_dcg_1 = np.sum(pr_dcg_1)
        pr_dcg_2 = r2 * (1. / np.log2(np.arange(2, topk + 2)))
        pr_dcg_2 = np.sum(pr_dcg_2)
        ex1_sum+=ex_dcg_1
        ex2_sum+=ex_dcg_2
        dcg1_sum+=pr_dcg_1
        dcg2_sum+=pr_dcg_2
        group_1_sum+=ex_n_1
        group_2_sum+=ex_n_2
    ex1_sum=ex1_sum/group_1_sum
    ex2_sum=ex2_sum/group_2_sum
    dcg1_sum=dcg1_sum/group_1_sum
    dcg2_sum=dcg2_sum/group_2_sum
    result=(ex1_sum/dcg1_sum)/(ex2_sum/dcg2_sum)
    return result