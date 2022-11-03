import os
import world
import dataloader
import model
import utils
import Procedure
import sample
from pprint import pprint


data_path = world.DATA_PATH+'/'+world.dataset+'/'+world.datasetpkl+'.pkl'
# if world.ONE:
#     # data_path = data_path + "_one"
#     print("{leave-one-out}:", data_path)
# else:
#     dataset = dataloader.Loader(path=data_path)
dataset = dataloader.Loader(path=data_path)
if world.DISTILL:
    print('===========DISTILL================')
    pprint(world.config)
    # print("beta:", world.beta)
    print("DNS K:", world.DNS_K)
    print("sample methods:", world.SAMPLE_METHOD)
    print("comment:", world.comment)
    print("Test Topks:", world.topks)
    print("kd weight:", world.kd_weight)
    print('===========end===================')
else:
    print('===========config================')
    pprint(world.config)
    print("cores for test:", world.CORES)
    print("comment:", world.comment)
    print("Weight path:", world.PATH)
    print("Test Topks:", world.topks)
    print("Train Method:", world.method)
    if world.method == 'dns':
        print(">>DNS K:", world.DNS_K)
    print("using bpr loss")
    print('===========end===================')

MODELS = {
    'BPRMF':model.BPRMF,
    'BPRMFExpert':model.BPRMFExpert,
    'LightGCN':model.LightGCN,
    'HTD':model.HTD,
    'PDA':model.PDA,
    'PDALightGCN':model.PDALightGCN,

}

TRAIN = {
    'category': Procedure.BPR_train_DNS_neg_cate,
    'original': Procedure.BPR_train_DNS_neg,
}

DISTILL_TRAIN = {
    'batch': Procedure.Distill_DNS_yield,
    'epoch': Procedure.Distill_DNS,
}

SAMPLER = {
    'combine' : sample.DistillSample,
    'RD'     : sample.RD,
    'CD'     : sample.CD,
    'DE_RRD'     : sample.RRD,
    'HTD'     : sample.HTD,
    'UnKD'     : sample.UnKD,

}