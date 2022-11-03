'''
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
import torch
from enum import Enum
from parse import parse_args
import multiprocessing
import sys



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ROOT_PATH = '..'
CODE_PATH = os.path.join(ROOT_PATH, 'code')
FILE_PATH = os.path.join(CODE_PATH, 'checkpoints')
BOARD_PATH = os.path.join(CODE_PATH, 'runs')
#DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATA_PATH = ROOT_PATH+'/data'
LOG_PATH = os.path.join(ROOT_PATH, 'log')


sys.path.append(os.path.join(CODE_PATH, 'sources'))

args = parse_args()
ARGS = args
EMBEDDING = args.embedding
DE_EMBEDDING=args.de_loss
SAMPLE_METHOD = args.sampler
# print(SAMPLE_METHOD)
# CD = False
#os.environ['CUDA_VISIBLE_DEVICE']=args.GPU
#print(torch.cuda.current_device())
config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon']
all_models = ['mf','BPRMF']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['test_u_batch_size'] = args.testbatch
config['lr'] = args.lr
config['decay'] = args.decay
config['teacher_dim'] = args.teacher_dim
config['teacher_layer'] = args.teacher_layer
config['teacher_model'] = args.teacher_model
config['num_expert'] = args.num_expert
config['de_loss'] = args.de_loss
config['de_weight'] = args.de_weight
config['add_weight'] = args.add_weight
config['kd_weight'] = args.kd_weight
config['keep_prob']  = args.keepprob
config['A_split'] = False
config['bigdata'] = False
config['pretrain'] = args.pretrain
config['dropout'] = args.dropout
config['decay_pcc'] = args.decay_pcc
config['lamda'] = args.lamda
config['alpha'] = args.alpha
config['split_group'] = args.split_group
config['sample_num'] = args.sample_num
de_weight=args.de_weight
DNS_K = args.dns_k
margin=args.margin
method = args.method
distill_method=args.distill_method
CORES = multiprocessing.cpu_count() // 2
SEED = args.seed
dataset = args.dataset
datasetpkl=args.datasetpkl
model_name = args.model
teacher_model_name=args.teacher_model

# else:
#     model_method = 'lgn'
# if dataset not in all_dataset:
# raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")

TRAIN_epochs = args.epochs
PATH = args.path
T = args.T
beta = args.beta
p0 = args.p0
startepoch = args.startepoch
stopflag=args.stopflag
topks = eval(args.topks)
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str, ends='\n'):
    print(f"\033[0;30;43m{words}\033[0m", end=ends)

