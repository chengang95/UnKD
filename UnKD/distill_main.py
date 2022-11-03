'''
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Distill model
'''
import os
import time
import world
import utils
import torch
import Procedure
import numpy as np
from world import cprint
from pprint import pprint
from tensorboardX import SummaryWriter
from sample import DistillSample
import tracemalloc

tracemalloc.start()
# ----------------------------------------------------------------------------
# global
world.DISTILL = True
if len(world.comment) == 0:
    comment = f"{world.SAMPLE_METHOD}"
    if world.EMBEDDING:
        comment = comment+"-embed"
    world.comment = comment
# ----------------------------------------------------------------------------
# set seed
utils.set_seed(world.SEED)
print(f"[SEED:{world.SEED}]")
# ----------------------------------------------------------------------------
# init model
import register
from register import dataset

# ----------------------------------------------------------------------------
# loading teacher
teacher_file = utils.getFileName(world.teacher_model_name,
                         world.dataset,
                         world.config['teacher_dim'],
                         layers=world.config['teacher_layer'],
                         dns_k=world.DNS_K)
teacher_file = str(world.de_weight)+'-'+teacher_file
teacher_file = str(world.t_lambda_pop)+'-'+teacher_file
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
if world.SAMPLE_METHOD=='DE_RRD':
    student_model = register.MODELS['lep'](world.config, dataset, teacher_model)
elif world.SAMPLE_METHOD=='SD':
    student_model = register.MODELS['newModel'](world.config, dataset, teacher_model)
else:
    student_model = register.MODELS[world.model_name](world.config, dataset)

# ----------------------------------------------------------------------------
# to device
student_model = student_model.cuda()
teacher_model = teacher_model.cuda()
# ----------------------------------------------------------------------------
# choosing paradigms
procedure = register.DISTILL_TRAIN['epoch']
sampler = register.SAMPLER[world.SAMPLE_METHOD](dataset, student_model, teacher_model, world.DNS_K)

bpr = utils.BPRLoss(student_model, world.config)
# ----------------------------------------------------------------------------
# get names
file = utils.getFileName(world.model_name, 
                         world.dataset,
                         world.config['latent_dim_rec'], 
                         layers=world.config['lightGCN_n_layers'],
                         dns_k=world.DNS_K)
file = world.teacher_model_name+'-'+world.SAMPLE_METHOD+'-'+str(world.config['teacher_dim'])+'-'+str(world.kd_weight)+'-'+str(world.config['de_weight'])+'-'+str(world.lambda_pop)+ '-' + file
weight_file = os.path.join(world.FILE_PATH, file)
print('-------------------------')
print(f"load and save student to {weight_file}")
if world.LOAD:
    utils.load(student_model, weight_file)
# ----------------------------------------------------------------------------
# training setting
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
        os.path.join(
            world.BOARD_PATH,time.strftime("%m-%d-%Hh-%Mm-") + f"{world.method}-{str(world.DNS_K)}-{file.split('.')[0]}-{world.comment}-DISTILL"
            )
        )
else:
    w = None
    world.cprint("not enable tensorflowboard")
earlystop = utils.EarlyStop(patience=20,
                            model=student_model,
                            filename=weight_file)
# ----------------------------------------------------------------------------
# test teacher
cprint("[TEST Teacher]")
results = Procedure.Test(dataset, teacher_model, 0, None, world.config['multicore'], valid=False)
pprint(results)
# ----------------------------------------------------------------------------
# start training
try:
    for epoch in range(world.TRAIN_epochs):

        start = time.time()
        output_information = procedure(dataset, student_model, sampler, bpr, epoch, w=w)

        print(
            f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {output_information}'
        )
        # snapshot = tracemalloc.take_snapshot()
        # utils.display_top(snapshot)
        print(f"    [TEST TIME] {time.time() - start}")
        if epoch %5 == 0:
            start = time.time()
            cprint("    [TEST]")
            results = Procedure.Test(dataset, student_model, epoch, w, world.config['multicore'], valid=True)
            pprint(results)
            print(f"    [TEST TIME] {time.time() - start}")
            if earlystop.step(epoch,results):
                print("trigger earlystop")
                print(f"best epoch:{earlystop.best_epoch}")
                print(f"best results:{earlystop.best_result}")
                break
finally:
    if world.tensorboard:
        w.close()

best_result = earlystop.best_result
torch.save(earlystop.best_model, weight_file)
student_model.load_state_dict(earlystop.best_model)
results = Procedure.Test(dataset,
                         student_model,
                         world.TRAIN_epochs,
                         valid=False)
log_file = os.path.join(world.LOG_PATH, utils.getLogFile())
with open(log_file, 'a') as f:
    f.write("#######################################\n")
    f.write(f"SEED: {world.SEED}, DNS_K: {str(world.DNS_K)}, Stop at: {earlystop.best_epoch+1}/{world.TRAIN_epochs}\n"\
            f"flag: {file.split('.')[0]}. \nLR: {world.config['lr']}, DECAY: {world.config['decay']}\n"\
            f"TopK: {world.topks}\n")
    f.write(f"%%Valid%%\n{best_result}\n%%TEST%%\n{results}\n")
    f.close()