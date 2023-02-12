# -*- coding: UTF-8 -*-

from __future__ import print_function
import os

from torch import optim

from sample import Sample_neg_cate
from model import OneLinear
from world import cprint

import Procedure

print('*** Current working path ***')
print(os.getcwd())
# os.chdir(os.getcwd()+"/NeuRec")

import os

import time
import multiprocessing

import world
import torch
import numpy as np
import utils




import sys
sys.path.append(os.getcwd())
print(sys.path)
import random as rd
import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def term(sig_num, addtion):
    print('term current pid is %s, group id is %s' % (os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
signal.signal(signal.SIGTERM, term)

cores = multiprocessing.cpu_count() // 2
max_pre = 1
print("half cores:",cores)
# ----------------------------------------------------------------------------
utils.set_seed(world.SEED)
print(f"[SEED:{world.SEED}]")
# ----------------------------------------------------------------------------
# init model
world.DISTILL = False
if len(world.comment) == 0:
    comment = f"{world.method}"
    if world.EMBEDDING:
        comment = comment + "-embed"
    world.comment = comment
import register
from register import dataset



if __name__ == '__main__':

    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    procedure = register.TRAIN[world.method]
    opt_loss = optim.Adam(Recmodel.parameters(), lr=world.config['lr'])

    bpr = utils.BPRLoss(Recmodel, world.config, opt_loss)
    # ----------------------------------------------------------------------------
    file = utils.getFileName(world.model_name,
                             world.datasetpkl,
                             world.config['latent_dim_rec'],
                             layers=world.config['lightGCN_n_layers'])
    file=world.comment+'-'+str(world.config['lr'])+'-'+str(world.config['decay'])+'-'+str(world.config['decay_pcc'])+'-'+file
    weight_file = os.path.join(world.FILE_PATH, file)
    print(f"load and save to {weight_file}")
    # ----------------------------------------------------------------------------
    earlystop = utils.EarlyStop(patience=20, model=Recmodel, filename=weight_file)
    Recmodel = Recmodel.cuda()
    totalparam = 0
    for param in Recmodel.parameters():
        mltbalue = np.prod(param.size())
        totalparam += mltbalue
    print(f'total params:{totalparam}')
    # start training
    best_div=None
    for epoch in range(world.TRAIN_epochs):

        start = time.time()
        output_information,aver_loss = procedure(dataset, Recmodel, bpr, (epoch+1),None)

        if (epoch+1) % 5 == 0 or (epoch+1)==world.TRAIN_epochs:
            print(
                f'EPOCH[{(epoch + 1)}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {output_information}'
            )
            cprint("Valid", ends=': ')
            results,other_out= Procedure.Test(dataset,Recmodel,(epoch+1),valid=True)
            print(results)
            if earlystop.step((epoch+1), results,other_out):
                print("trigger earlystop")
                print(f"best epoch:{earlystop.best_epoch}")
                print(f"best results:{earlystop.best_result}")
                break


    utils.load(Recmodel, weight_file)
    best_result, other_out = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, valid=True)
    results,other_out = Procedure.Test(dataset,Recmodel,world.TRAIN_epochs,valid=False)
    allresults = Procedure.TestAll(dataset, other_out['rankdict'], world.TRAIN_epochs, valid=False)
    resultBygroup10=utils.resultBygroup(other_out['rankdict'],dataset,10)
    resultBygroup20 = utils.resultBygroup(other_out['rankdict'], dataset, 20)

    log_file = os.path.join(world.LOG_PATH, utils.getLogFile())
    with open(log_file, 'a') as f:
        f.write("#######################################\n")
        f.write(f"{file}\n")
        f.write(f"SEED: {world.SEED}, DNS_K: {str(world.DNS_K)}, Stop at: {earlystop.best_epoch}/{world.TRAIN_epochs}\n"\
                f"flag: {file.split('.')[0]}. \nLR: {world.config['lr']}, DECAY: {world.config['decay']}\n"\
                f"TopK: {world.topks}\n")
        f.write(f"%%Valid%%\n{best_result}\n%%TEST%%\n{results}\n")
        f.write(f"%%allResult%%\n{allresults}\n")
        f.write(f"{resultBygroup10}\n{resultBygroup20}\n")
        f.close()






