'''
Design training and test process
'''
import os
import world
import torch
import utils
import model
import dataloader
import multiprocessing
import numpy as np
from time import time
from pprint import pprint

from dpp import dpp_sw, dpp
from sample import DistillSample, Sample_DNS_cate_python, Sample_DNS_cate_epoch, Sample_DNS_double_cate_epoch
# from sample import DistillLogits
from model import PairWiseModel, BasicModel
from utils import time2str, timer

from sample import Sample_DNS_python,Sample_original


item_count = None

CORES = multiprocessing.cpu_count() // 2


def Distill_DNS_yield(dataset, student, sampler, loss_class, epoch, w=None):
    """Batch version of Distill_DNS, using a generator to offer samples
    """
    bpr: utils.BPRLoss = loss_class
    student.train()
    aver_loss = 0
    with timer(name='sampling'):
        S = sampler.PerSample(batch=world.config['bpr_batch_size'])
    total_batch = dataset.trainDataSize // world.config['bpr_batch_size'] + 1
    for batch_i, Pairs in enumerate(S):
        Pairs = torch.from_numpy(Pairs).long().cuda()
        # print(Pairs.shape)
        batch_users, batch_pos, batch_neg = Pairs[:, 0], Pairs[:, 1], Pairs[:,
                                                                            2:]
        with timer(name="KD"):
            KD_loss = sampler.Sample(
                batch_users, batch_pos, batch_neg, epoch)
        with timer(name="BP"):
            cri = bpr.stageOne(batch_users,
                               batch_pos,
                               batch_neg,
                               epoch)
        aver_loss += cri

        del Pairs
    aver_loss = aver_loss / total_batch
    info = f"{timer.dict()}[BPR loss{aver_loss:.3e}]"
    timer.zero()
    return info



def BPR_train_DNS_neg(dataset, recommend_model, loss_class, epoch,S):
    """Traininf procedure for DNS algorithms

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        recommend_model (PairWiseModel): recommend model with small dim
        loss_class (utils.BPRLoss): class to get BPR training loss, and BackPropagation
        epoch (int):
        w (SummaryWriter, optional): Tensorboard writer

    Returns:
        str: summary of aver loss and running time for one epoch
    """
    Recmodel: PairWiseModel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    S = Sample_original(dataset)
    S = torch.Tensor(S).long().cuda()
    users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2]
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg,epoch)
        aver_loss += cri
    # print(f"DNS[sampling][{time()-DNS_time:.1f}={DNS_time1:.2f}+{DNS_time2:.2f}]")
    aver_loss = aver_loss / total_batch
    return f"{timer.dict()}[BPR[aver loss{aver_loss:.3e}]",aver_loss

def BPR_train_DNS_neg_cate(dataset, recommend_model, loss_class, epoch,S):
    """Traininf procedure for DNS algorithms

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        recommend_model (PairWiseModel): recommend model with small dim
        loss_class (utils.BPRLoss): class to get BPR training loss, and BackPropagation
        epoch (int):
        w (SummaryWriter, optional): Tensorboard writer

    Returns:
        str: summary of aver loss and running time for one epoch
    """
    print('category')
    Recmodel: PairWiseModel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    S = Sample_original(dataset)
    S = torch.Tensor(S).long().cuda()
    users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2]
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, None,epoch)
        aver_loss += cri
        cri = bpr.stageOne(batch_users, None, batch_neg, epoch)
        aver_loss += cri
    # print(f"DNS[sampling][{time()-DNS_time:.1f}={DNS_time1:.2f}+{DNS_time2:.2f}]")
    aver_loss = aver_loss / total_batch
    return f"{timer.dict()}[BPR[aver loss{aver_loss:.3e}]",aver_loss


def Distill_DNS(dataset, student, sampler, loss_class, epoch, w=None):
    """Training procedure for distillation methods

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        student (PairWiseModel): recommend model with small dim
        sampler (DS|DL|RD|CD): tons of distill methods defined in sample.py
        loss_class (utils.BPRLoss): class to get BPR training loss, and BackPropagation
        epoch (int):
        w (SummaryWriter, optional): Tensorboard writer

    Returns:
        str: summary of aver loss and running time for one epoch
    """
    bpr: utils.BPRLoss = loss_class
    student.train()
    aver_loss = 0
    aver_kd = 0
    with timer(name='sampling'):
        S = sampler.PerSample(epoch)
    S = torch.Tensor(S).long().cuda()
    users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2]
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        with timer(name="KD"):
            KD_loss = sampler.Sample(
                batch_users, batch_pos, batch_neg, epoch)
        with timer(name="BP"):
            cri = bpr.stageOne(batch_users,batch_pos,batch_neg,epoch,KD_loss)
        aver_loss += cri
        aver_kd+=KD_loss.cpu().item()
        # Additional section------------------------
        #
        # ------------------------------------------
    aver_loss = aver_loss / total_batch
    aver_kd = aver_kd / total_batch
    info = f"{timer.dict()}[KD loss{aver_kd:.3e}][BPR loss{aver_loss:.3e}]"
    timer.zero()
    return info


def Distill_DNS_BD(dataset, student,teacher, sampler, bpr_T,bpr_S, epoch, w=None):
    """Training procedure for distillation methods

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        student (PairWiseModel): recommend model with small dim
        sampler (DS|DL|RD|CD): tons of distill methods defined in sample.py
        loss_class (utils.BPRLoss): class to get BPR training loss, and BackPropagation
        epoch (int):
        w (SummaryWriter, optional): Tensorboard writer

    Returns:
        str: summary of aver loss and running time for one epoch
    """
    bpr_T: utils.BPRLoss = bpr_T
    bpr_S: utils.BPRLoss = bpr_S
    student.train()
    teacher.train()
    aver_loss = 0
    aver_kd = 0
    with timer(name='sampling'):
        S = sampler.PerSample(epoch)
    S = torch.Tensor(S).long().cuda()
    users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2]
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        bpr_S.opt.zero_grad()
        bpr_T.opt.zero_grad()
        with timer(name="KD"):
            KD_loss_T,KD_loss_S = sampler.Sample(
                batch_users, batch_pos, batch_neg, epoch)
        with timer(name="BP"):
            cri = bpr_S.stageTwo(batch_users,batch_pos,batch_neg,epoch,KD_loss_S)
            bpr_T.stageTwo(batch_users, batch_pos, batch_neg, epoch, KD_loss_T)
        aver_loss += cri
        if epoch>=100:
            aver_kd+=KD_loss_S.cpu().item()
        # Additional section------------------------
        #
        # ------------------------------------------
    aver_loss = aver_loss / total_batch
    aver_kd = aver_kd / total_batch
    info = f"{timer.dict()}[KD loss{aver_kd:.3e}][BPR loss{aver_loss:.3e}]"
    timer.zero()
    return info

# ******************************************************************************
# ============================================================================**
# ============================================================================**
# ******************************************************************************
# TEST
def test_one_batch(X):
    """helper function for Test
    """
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg,recallbygroup = [], [], [],[]
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        #dcg.append(utils.NDCGatK_r_ONE(r, k))
    return {
        'recall': np.array(recall),
        'precision': np.array(pre),
        'ndcg': np.array(ndcg),
        #'dcg':np.array(dcg)
    }
def test_recallbygroup(X,dataset):
    """helper function for Test
    """
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    recallbygroup = []
    precisionbygroup = []
    for k in world.topks:
        #dcg.append(utils.NDCGatK_r_ONE(r, k))
        r1,r2=utils.RecallAndPrecisionByGroup(dataset,groundTrue,sorted_items,r, k)
        recallbygroup.append(r1)
        precisionbygroup.append(r2)
    return {'recallbygroup':recallbygroup,'precisionbygroup':precisionbygroup}

def test_one_batch_ONE(X):
    """helper function for Test, customized for leave-one-out dataset
    """
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel_ONE(groundTrue, sorted_items)
    ndcg, hr = [], []
    for k in world.topks:
        ndcg.append(utils.NDCGatK_r_ONE(r, k))
        hr.append(utils.HRatK_ONE(r, k))
    return {'ndcg': np.array(ndcg), 'hr': np.array(hr)}


def Test(dataset, Recmodel, epoch,valid=True):
    """evaluate models

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        Recmodel (PairWiseModel):
        epoch (int): 
        w (SummaryWriter, optional): Tensorboard writer
        multicore (int, optional): The num of cpu cores for testing. Defaults to 0.

    Returns:
        dict: summary of metrics
    """
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict
    # eval mode with no dropout
    Recmodel.eval()
    max_K = max(world.topks)
    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        rankdict = {}
        truedict = {}
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -1e10
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            for i in range(len(rating_K)):
                rankdict[batch_users[i].item()]=rating_K[i].cpu().numpy()
                truedict[batch_users[i].item()] = groundTrue[i]
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        results = {
            'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks)),
            #'dcg': np.zeros(len(world.topks))
            #'recallbygroup': np.zeros((len(world.topks),2)),
        }
        pre_results = []
        recallbygroup=[]
        for x in X:
            pre_results.append(test_one_batch(x))
            #recallbygroup.append(test_recallbygroup(x,dataset))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        other_out={}
        other_out['rankdict'] = rankdict
        other_out['truedict'] = truedict
        # for result in recallbygroup:
        #     results['recallbygroup'] += result
        # results['recallbygroup'] /= float(len(users))
        return results,other_out


def TestFair(dataset, Recmodel, epoch,valid=True):
    """evaluate models

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        Recmodel (PairWiseModel):
        epoch (int):
        w (SummaryWriter, optional): Tensorboard writer
        multicore (int, optional): The num of cpu cores for testing. Defaults to 0.

    Returns:
        dict: summary of metrics
    """
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict
    # eval mode with no dropout
    Recmodel.eval()
    max_K = max(world.topks)
    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        rankdict = {}
        truedict = {}
        ratingdict = {}
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -1e10
            rating_used, rating_K = torch.topk(rating, k=20)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            for i in range(len(rating_K)):
                rankdict[batch_users[i].item()]=rating_K[i].cpu().numpy()
                truedict[batch_users[i].item()] = groundTrue[i]
                ratingdict[batch_users[i].item()]=rating_used[i].cpu()

        other_out={}
        other_out['rankdict'] = rankdict
        other_out['truedict'] = truedict
        other_out['ratingdict'] = ratingdict
        return other_out

def TestAll(dataset, rankDict, epoch,valid=True):
    """evaluate models

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        Recmodel (PairWiseModel):
        epoch (int):
        w (SummaryWriter, optional): Tensorboard writer
        multicore (int, optional): The num of cpu cores for testing. Defaults to 0.

    Returns:
        dict: summary of metrics
    """
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict
    # eval mode with no dropout
    max_K = max(world.topks)
    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            groundTrue = [testDict[u] for u in batch_users]
            rating = torch.Tensor([rankDict[u] for u in batch_users])
            users_list.append(batch_users)
            groundTrue_list.append(groundTrue)
            rating_list.append(rating)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        results = {
            'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks)),
            #'dcg': np.zeros(len(world.topks))
            'recallbygroup': np.zeros((len(world.topks),len(dataset.item_group))),
            'precisionbygroup': np.zeros((len(world.topks), len(dataset.item_group))),
        }
        pre_results = []
        recallbygroup=[]
        for x in X:
            pre_results.append(test_one_batch(x))
            recallbygroup.append(test_recallbygroup(x,dataset))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        for result in recallbygroup:
            results['recallbygroup'] += result['recallbygroup']
            results['precisionbygroup'] += result['precisionbygroup']
        results['recallbygroup'] /= float(len(users))
        results['precisionbygroup'] /= float(len(users))
        return results


