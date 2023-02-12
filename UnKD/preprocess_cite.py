import math

import numpy
import numpy as np
import scipy.sparse as sp
import pickle
import os
import pandas as pd
from _collections import defaultdict
import random
from sklearn import preprocessing

import world


def load_interaction(data_tain):
    all_data = []
    user_src2tgt_dict, item_src2tgt_dict = {}, {}
    user_count={}
    item_count = {}
    tag_c={}
    f = open(data_tain, 'r')
    u=0
    for eachline in f:
        # eachline = json.loads(eachline)
        # u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        eachline = eachline.strip().split(' ')
        items =eachline[1:]
        items=[int(i) for i in items]
        for i in items:
            if (u,i) not in tag_c.keys():
                if i not in item_count:
                    item_count[i]=0
                item_count[i]+=1
                tag_c[(u,i)]=1
        u=u+1
    f.close()

    tag_c = {}
    u=0
    f = open(data_tain, 'r')
    for eachline in f:
        # eachline = json.loads(eachline)
        # u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        eachline = eachline.strip().split(' ')
        items = eachline[1:]
        items = [int(i) for i in items]
        for i in items:
            if (u, i) not in tag_c.keys() :
                if item_count[i]>=1:
                    if u not in user_count:
                        user_count[u] = 0
                    user_count[u] += 1
                    tag_c[(u, i)] = 1
        u = u + 1
    f.close()

    tag_c = {}
    u=0
    f = open(data_tain, 'r')
    for eachline in f:
        #eachline = json.loads(eachline)
        #u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        eachline = eachline.strip().split(' ')
        items=eachline[1:]
        items = [int(i) for i in items]
        for i in items:
            if (u, i) not in tag_c.keys():
                if  u in user_count and user_count[u]>=5 :
                    if u not in user_src2tgt_dict:
                        user_src2tgt_dict[u] = len(user_src2tgt_dict)
                    user = user_src2tgt_dict[u]
                    if i not in item_src2tgt_dict:
                        item_src2tgt_dict[i] = len(item_src2tgt_dict)
                    item = item_src2tgt_dict[i]
                    all_data.append([user, item, 1])
                    tag_c[(u,i)]=1
        u = u + 1
    return all_data, user_src2tgt_dict, item_src2tgt_dict

def create_user_list(all_data, user_size):
    user_list = [dict() for u in range(user_size)]
    count = 0
    for u, i, t in all_data:
        user_list[u][i] = t
    return user_list




def split_train_test(user_list, test_size=0.1, time_order=False):
    train_user_list = [None] * len(user_list)
    test_user_list = [None] * len(user_list)
    count1=0
    for user, item_dict in enumerate(user_list):
        if time_order:
            # Choose latest item
            item = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
            latest_item = item[:math.ceil(len(item)*(test_size))]
            assert max(item_dict.values()) == latest_item[0][1]
            test_item = set(map(lambda x: x[0], latest_item))
            #valid_item = set(map(lambda x: x[0], latest_item[int(len(latest_item) * 0.5):]))
            # valid_item=set(np.random.choice(list(test_item),
            #                      size=int(len(test_item) * 0.5),
            #                      replace=False))
            # test_item=test_item-valid_item
        else:
            # Random select
            test_item = set(np.random.choice(list(item_dict.keys()),
                                             size=math.ceil(len(item_dict)*(test_size)),
                                             replace=False))
            # valid_item = set(np.random.choice(list(test_item),
            #                                   size=int(len(test_item) * 0.5),
            #                                   replace=False))
            #test_item = test_item - valid_item

        #if user>0:
        #    assert (len(test_item) > 0), "No test item for user %d" % user
        if  len(test_item)==0:
            # print("No test item for user %d" % user)
            count1+=1
        test_user_list[user] = list(test_item)
        train_user_list[user] = list(set(item_dict.keys()) - test_item)

    sampel_user=np.random.choice(range(len(user_list)), size=int(len(user_list)*0.2), replace=False)
    test_user_dict={}
    valid_user_dict={}
    for u in range(len(test_user_list)):
        if u in sampel_user:
            valid_user_dict[u]=test_user_list[u]
        else:
            test_user_dict[u]=test_user_list[u]
    print("the number of user of no test item   %d" % count1)
    return train_user_list, test_user_dict,valid_user_dict









if __name__ == '__main__':
    random.seed(world.SEED)
    np.random.seed(world.SEED)
    test_ratio = 0.2

    f_train = world.DATA_PATH+'/cite/users.dat'
    f_out = world.DATA_PATH+'/cite/cite@20230212'+'.pkl'

    #load interaction
    all_data, user_src2tgt_dict, item_src2tgt_dict = load_interaction(f_train)
    user_size = len(user_src2tgt_dict)
    item_size = len(item_src2tgt_dict)

    total_user_list = create_user_list(all_data, user_size)  # 把[u,i,r]转为list[u]={item...}
    print('user_size, item_size,inter_num:', user_size, item_size, len(all_data))

    # ----------------------------------partition
    train_user_list, test_user_dict,valid_user_dict= split_train_test(total_user_list,  # 划分train/test
                                                           test_size=test_ratio,time_order=False)
    #train_user_list=deal_train_list(train_user_list,total_user_list,user_size,item_cate_dict_s,current_cate_list)
    print('min len(user_itemlist): ', min([len(ilist) for ilist in train_user_list]))
    print('train avg. items per user: ', np.mean([len(u) for u in train_user_list]))  # 165.57
    print('test avg. items per user: ', np.mean([len(test_user_dict[u]) for u in test_user_dict.keys()]))
    print('valid avg. items per user: ', np.mean([len(valid_user_dict[u]) for u in valid_user_dict.keys()]))
    print('Complete creating pair')

    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_dict': test_user_dict,'valid_user_dict':valid_user_dict
               }
    with open(f_out, 'wb') as f:
        #json.dump(dataset, f)
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

#[4,7,0]