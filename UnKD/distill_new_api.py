# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
from pprint import pprint

from torch import optim

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
    # loading teacher
    teacher_file = utils.getFileName(world.teacher_model_name,
                                     world.datasetpkl,
                                     world.config['teacher_dim'],
                                     layers=world.config['teacher_layer'])
    teacher_file = 'div_20220530_ndcg_lab'+'-'+str(0.001)+'-'+str(0.001)+'-'+str(0.0)+'-'+teacher_file
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
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # loading student
    world.cprint('student')
    if world.SAMPLE_METHOD == 'DE_RRD':
        student_model = register.MODELS['BPRMFExpert'](world.config, dataset, teacher_model)
    elif world.SAMPLE_METHOD == 'HTD':
        student_model = register.MODELS['HTD'](world.config, dataset, teacher_model)
    else:
        student_model = register.MODELS[world.model_name](world.config, dataset)

    # ----------------------------------------------------------------------------
    # to device
    student_model = student_model.cuda()
    teacher_model = teacher_model.cuda()
    # test teacher
    cprint("[TEST Teacher]")
    results, other_out = Procedure.Test(dataset, teacher_model, 0, valid=False)
    pprint(results)

    # ----------------------------------------------------------------------------
    # choosing paradigms
    print(world.distill_method)
    procedure = Procedure.Distill_DNS
    sampler = register.SAMPLER[world.SAMPLE_METHOD](dataset, student_model, teacher_model, world.DNS_K)
    opt_loss = optim.Adam(student_model.parameters(), lr=world.config['lr'])
    bpr = utils.BPRLoss(student_model, world.config, opt_loss)

    # ------------------
    # ----------------------------------------------------------
    # get names
    file = utils.getFileName(world.model_name,
                             world.datasetpkl,
                             world.config['latent_dim_rec'],
                             layers=world.config['lightGCN_n_layers'])
    file = world.comment+'-'+world.teacher_model_name + '-'+ world.SAMPLE_METHOD + '-' + str(world.config['teacher_dim']) + '-' \
           + str(world.config['kd_weight']) + '-' + str(world.config['lr']) +'-'+ str(world.config['decay']) +'-'+file
    weight_file = os.path.join(world.FILE_PATH, file)
    print('-------------------------')
    print(f"load and save student to {weight_file}")
    # ----------------------------------------------------------------------------
    # training setting
    earlystop = utils.EarlyStop(patience=20,
                                model=student_model,
                                filename=weight_file)
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # start training

    for epoch in range(world.TRAIN_epochs):

        start = time.time()
        output_information = procedure(dataset, student_model, sampler, bpr, (epoch+1))

        if (epoch+1) % 5 == 0 or (epoch+1)==world.TRAIN_epochs:
            print(
                f'EPOCH[{(epoch + 1)}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {output_information}'
            )
            cprint("[valid]")
            results,otrher_out= Procedure.Test(dataset, student_model, (epoch+1), valid=True)
            pprint(results)
            print(f"    [TEST TIME] {time.time() - start}")
            if earlystop.step((epoch+1), results,otrher_out):
                print("trigger earlystop")
                print(f"best epoch:{earlystop.best_epoch}")
                print(f"best results:{earlystop.best_result}")
                break

    utils.load(student_model, weight_file)
    best_result, other_out = Procedure.Test(dataset, student_model, world.TRAIN_epochs, valid=True)
    results, other_out = Procedure.Test(dataset, student_model, world.TRAIN_epochs, valid=False)
    allresults = Procedure.TestAll(dataset, other_out['rankdict'], world.TRAIN_epochs, valid=False)
    resultBygroup10 = utils.resultBygroup(other_out['rankdict'], dataset, 10)
    resultBygroup20 = utils.resultBygroup(other_out['rankdict'], dataset, 20)

    log_file = os.path.join(world.LOG_PATH, utils.getLogFile())
    with open(log_file, 'a') as f:
        f.write("#######################################\n")
        f.write(f"{file}\n")
        f.write(f"SEED: {world.SEED}, DNS_K: {str(world.DNS_K)}, Stop at: {earlystop.best_epoch}/{world.TRAIN_epochs}\n" \
                f"flag: {file.split('.')[0]}. \nLR: {world.config['lr']}, DECAY: {world.config['decay']}\n" \
                f"TopK: {world.topks}\n")
        f.write(f"%%Valid%%\n{best_result}\n%%TEST%%\n{results}\n")
        f.write(f"%%allResult%%\n{allresults}\n")
        f.write(f"{resultBygroup10}\n{resultBygroup20}\n")
        f.close()






