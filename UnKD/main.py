#!/opt/conda/bin/python
'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Main program
'''
import os
import time
import world
import utils
import torch
import Procedure
import numpy as np
from pprint import pprint
from world import cprint
import sample
from tensorboardX import SummaryWriter

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

# if world.EMBEDDING:
#     # embedding distillation
#     print("distill")
#     tea_config = utils.getTeacherConfig(world.config)
#     world.cprint('teacher')
#     teacher_model = register.MODELS[world.model_name](tea_config,
#                                                       dataset,
#                                                       fix=True)
#     teacher_model.eval()
#     teacher_file = utils.getFileName(world.model_name,
#                                      world.dataset,
#                                      world.config['teacher_dim'],
#                                      layers=world.config['teacher_layer'])
#     teacher_weight_file = os.path.join(world.FILE_PATH, teacher_file)
#     print('-------------------------')
#     world.cprint("loaded teacher weights from")
#     print(teacher_weight_file)
#     print('-------------------------')
#     utils.load(teacher_model, teacher_weight_file)
#     teacher_model = teacher_model.cuda()
#     cprint("[TEST Teacher]")
#     results = Procedure.Test(dataset, teacher_model, 0, None,
#                              world.config['multicore'])
#     pprint(results)
#     Recmodel = register.MODELS['leb'](world.config, dataset, teacher_model)
#     print(Recmodel)
# else:
#     Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
procedure = register.TRAIN[world.method]
bpr = utils.BPRLoss(Recmodel, world.config)
# ----------------------------------------------------------------------------
file = utils.getFileName(world.model_name,
                         world.dataset,
                         world.config['latent_dim_rec'],
                         layers=world.config['lightGCN_n_layers'],
                         dns_k=world.DNS_K)
file=str(world.lambda_pop)+'-'+str(world.de_weight)+'-'+file
weight_file = os.path.join(world.FILE_PATH, file)
print(f"load and save to {weight_file}")
if world.LOAD:
    utils.load(Recmodel, weight_file)
# ----------------------------------------------------------------------------
earlystop = utils.EarlyStop(patience=20, model=Recmodel, filename=weight_file)
Recmodel = Recmodel.cuda()
# ----------------------------------------------------------------------------
# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        os.path.join(
            world.BOARD_PATH,
            time.strftime("%m-%d-%Hh-%Mm-") +
            f"{world.method}-{str(world.DNS_K)}-{file.split('.')[0]}-{world.comment}"
        ))
else:
    w = None
    world.cprint("not enable tensorflowboard")
# ----------------------------------------------------------------------------
# start training
try:
    for epoch in range(world.TRAIN_epochs):

        start = time.time()
        output_information = procedure(dataset, Recmodel, bpr, epoch, w=w)

        print(
            f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {output_information}'
        )
        if epoch % 5 == 0 and epoch != 0:
            cprint("TEST", ends=': ')
            results = Procedure.Test(dataset,
                                     Recmodel,
                                     epoch,
                                     w,
                                     world.config['multicore'],
                                     valid=True)
            print(results)
            if earlystop.step(epoch, results):
                print("trigger earlystop")
                print(f"best epoch:{earlystop.best_epoch}")
                print(f"best results:{earlystop.best_result}")
                break
finally:
    if world.tensorboard:
        w.close()

best_result = earlystop.best_result
torch.save(earlystop.best_model, weight_file)
Recmodel.load_state_dict(earlystop.best_model)
results = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, valid=False)
log_file = os.path.join(world.LOG_PATH, utils.getLogFile())
with open(log_file, 'a') as f:
    f.write("#######################################\n")
    f.write(f"SEED: {world.SEED}, DNS_K: {str(world.DNS_K)}, Stop at: {earlystop.best_epoch+1}/{world.TRAIN_epochs}\n"\
            f"flag: {file.split('.')[0]}. \nLR: {world.config['lr']}, DECAY: {world.config['decay']}\n"\
            f"TopK: {world.topks}\n")
    f.write(f"%%Valid%%\n{best_result}\n%%TEST%%\n{results}\n")
    f.close()
