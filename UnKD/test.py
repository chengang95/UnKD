# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
import random

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

    #Recmodel = register.MODELS[world.model_name](world.config, dataset)
    world.cprint('student')

    procedure = register.TRAIN[world.method]
    # ----------------------------------------------------------------------------
    file = utils.getFileName(world.model_name,
                             world.datasetpkl,
                             world.config['latent_dim_rec'],
                             layers=world.config['lightGCN_n_layers'])
    #file=world.comment+'-'+str(world.config['lr'])+'-'+str(world.config['decay'])+'-'+str(world.config['add_weight'])+'-'+file

    file =world.comment + '-' + world.teacher_model_name + '-' + world.SAMPLE_METHOD + '-' + str(
        world.config['teacher_dim']) + '-' \
           + str(world.config['kd_weight']) + '-' + str(world.config['lr']) + '-' + str(
        world.config['decay']) + '-' + file
    weight_file = os.path.join(world.FILE_PATH, file)
    print(f"load and save to {weight_file}")
    # ----------------------------------------------------------------------------
    if world.SAMPLE_METHOD == 'DE_RRD':
        teacher_file = utils.getFileName(world.teacher_model_name,
                                         world.datasetpkl,
                                         world.config['teacher_dim'],
                                         layers=world.config['teacher_layer'])
        teacher_file = 'div_20220530_ndcg_lab' + '-' + str(0.001) + '-' + str(0.001) + '-' + str(
            world.config['add_weight']) + '-' + teacher_file
        teacher_weight_file = os.path.join(world.FILE_PATH, teacher_file)
        print('-------------------------')
        world.cprint("loaded teacher weights from")
        print(teacher_weight_file)
        print('-------------------------')
        teacher_config = utils.getTeacherConfig(world.config)
        world.cprint('teacher')
        teacher_model = register.MODELS[world.teacher_model_name](teacher_config,
                                                                  dataset,
                                                                  fix=True)
        teacher_model.eval()
        utils.load(teacher_model, teacher_weight_file)
        Recmodel = register.MODELS['BPRMFExpert'](world.config, dataset, teacher_model)
        print("_-------------")
    elif world.SAMPLE_METHOD == 'HTD':
        teacher_file = utils.getFileName(world.teacher_model_name,
                                         world.datasetpkl,
                                         world.config['teacher_dim'],
                                         layers=world.config['teacher_layer'])
        teacher_file = 'div_20220530_ndcg_lab' + '-' + str(0.001) + '-' + str(0.001) + '-' + str(
            world.config['add_weight']) + '-' + teacher_file
        teacher_weight_file = os.path.join(world.FILE_PATH, teacher_file)
        print('-------------------------')
        world.cprint("loaded teacher weights from")
        print(teacher_weight_file)
        print('-------------------------')
        teacher_config = utils.getTeacherConfig(world.config)
        world.cprint('teacher')
        teacher_model = register.MODELS[world.teacher_model_name](teacher_config,
                                                                  dataset,
                                                                  fix=True)
        teacher_model.eval()
        utils.load(teacher_model, teacher_weight_file)
        Recmodel = register.MODELS['HTD'](world.config, dataset, teacher_model)
    else:
        Recmodel = register.MODELS[world.model_name](world.config, dataset)

    utils.load(Recmodel, weight_file)
    Recmodel = Recmodel.cuda()

    results, other_out = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, valid=False)
    print(results)
    result = Procedure.TestAll(dataset, other_out['rankdict'], 0, valid=False)
    print(result)
    print("DTR")
    other_out = Procedure.TestFair(dataset, Recmodel, world.TRAIN_epochs, valid=False)
    ex = utils.getDTR(other_out['rankdict'], other_out['ratingdict'], other_out['truedict'], dataset, 10)
    print(ex)
    ex = utils.getDTR(other_out['rankdict'], other_out['ratingdict'], other_out['truedict'], dataset, 20)
    print(ex)

    print("result")
    #r=utils.resultBygroup(other_out['rankdict'],dataset,10)
    r = utils.resultBygroup(dataset.testDict, dataset, 100)
    print(r)
    print('*******************************************')



