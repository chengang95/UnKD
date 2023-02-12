'''
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=10,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=0.001,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--decay_pcc', type=float, default=0.0,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--a_fold', type=int,default=50,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=1024,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='cite',
                        help="available datasets: [ml1m app cite]")
    parser.add_argument('--datasetpkl', type=str, default='cite@20220530',
                        help="available datasets")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10,20]",
                        help="@k test list")
    parser.add_argument('--comment', type=str,default="div_20220621_ndcg_computer")
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--model', type=str, default='BPRMF', help='rec-model, support [mf, lgn]')
    parser.add_argument('--teacher_model', type=str, default='BPRMF', help='tacher-model, support [mf, lgn,ulgn]')
    parser.add_argument('--method', type=str, default='original', help='train process [original, category]')
    parser.add_argument('--distill_method', type=str, default=None, help='train process ')
    parser.add_argument('--dns_k', type=int, default=1, help='The polynomial degree for DNS(Dynamic Negative Sampling)')
    parser.add_argument('--testweight', type=float, default=1)
    parser.add_argument('--teacher_dim', type=int, default=100, help='teacher\'s dimension')
    parser.add_argument('--teacher_layer', type=int, default=1, help='teacher\'s layer')
    parser.add_argument('--startepoch', type=int, default=1, help='The epoch to start distillation')
    parser.add_argument('--T', type=float, default=1.0, help='The temperature for teacher distribution')
    parser.add_argument('--beta', type=float, default=1e-4, help='The beta')
    parser.add_argument('--p0', type=float, default=1.0, help='The p0')
    parser.add_argument('--one', type=int, default=0, help='leave one out')
    parser.add_argument('--embedding', type=int, default=0, help='enable embedding distillation')
    parser.add_argument('--sampler', type=str, default='UnKD')
    parser.add_argument('--num_expert', type=int, default=5)
    parser.add_argument('--de_loss', type=int, default=0)
    parser.add_argument('--de_weight', type=float, default=0.001)
    parser.add_argument('--kd_weight', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=0.0)
    parser.add_argument('--margin', type=int, default=100)
    parser.add_argument('--stopflag', type=str, default='ndcg')
    parser.add_argument('--add_weight', type=float, default=0.0)
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--split_group', type=int, default=2)
    parser.add_argument('--sample_num', type=int, default=30)
    return parser.parse_args()