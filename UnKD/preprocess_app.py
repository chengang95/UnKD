import math
import os
import pickle
import argparse

from itertools import islice
import numpy as np
import pandas as pd
import random
import json
from collections import defaultdict
from sklearn import preprocessing

import world


def load_interaction(data_dir):
    all_data = []

    user_count = {}
    f = open(data_dir, 'r')
    for eachline in f:
        eachline = json.loads(eachline)
        u, i, r, t = eachline['reviewerID'], eachline['asin'], eachline['overall'], eachline['unixReviewTime']
        if r > 0:
            if u not in user_count:
                user_count[u] = 0
            user_count[u] += 1
    f.close()


    f = open(data_dir, 'r')
    user_src2tgt_dict, item_src2tgt_dict = {}, {}
    for eachline in f:
        eachline = json.loads(eachline)
        u, i, r,t = eachline['reviewerID'], eachline['asin'], eachline['overall'],eachline['unixReviewTime']
        if r > 0 and user_count[u] >= 20:
            if u not in user_src2tgt_dict:
                user_src2tgt_dict[u] = len(user_src2tgt_dict)
            user = user_src2tgt_dict[u]
            if i not in item_src2tgt_dict:
                item_src2tgt_dict[i] = len(item_src2tgt_dict)
            item = item_src2tgt_dict[i]
            all_data.append([user, item, t])
    return all_data, user_src2tgt_dict, item_src2tgt_dict

def load_meta(f_meta, item_src2tgt_dict):
    cate_src2tgt_dict = {}
    cate_src_count = defaultdict(int)
    item_cate_dict = {}
    f = open(f_meta, 'r', encoding='utf-8')
    for eachline in f:
        eachline = json.dumps(eval(eachline))
        eachline = json.loads(eachline)

        i, c = eachline['asin'], eachline['categories']
        if i in item_src2tgt_dict:
            temp_c=c[0]
            category= temp_c[1]
            if category not in cate_src2tgt_dict:
                cate_src2tgt_dict[category] = len(cate_src2tgt_dict)
            cate = cate_src2tgt_dict[category]
            cate_src_count[category] += 1
            item_cate_dict[item_src2tgt_dict[i]] = [cate]   #item_id -> cate_id

    item_cate_dict = {k: v for k, v in sorted(item_cate_dict.items(), key=lambda kv: (kv[0], kv[1]))}
    print(cate_src_count)
    return cate_src2tgt_dict, item_cate_dict

def create_user_list(all_data, user_size):
    user_list = [dict() for u in range(user_size)]
    count = 0
    for u,i,w in all_data:
        user_list[u][i] = w
        count += 1
    print('interaction num:', count)
    return user_list

def split_train_test(user_list, test_size=0.2, time_order=False):
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
            # Random sappt
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
            print("No test item for user %d" % user)
            count1+=1
        test_user_list[user] = list(test_item)
        train_user_list[user] = list(set(item_dict.keys()) - test_item)

    sampel_user=np.random.choice(range(len(user_list)), size=int(len(user_list)*0.5), replace=False)
    test_user_dict={}
    valid_user_dict={}
    for u in range(len(test_user_list)):
            if u in sampel_user:
                test_user_dict[u]=test_user_list[u]
                #valid_user_dict[u]=test_user_list[u]
            else:
                valid_user_dict[u]=test_user_list[u]
                #test_user_dict[u] = test_user_list[u]
    print("the number of user of no test item   %d" % count1)
    return train_user_list, test_user_dict,valid_user_dict









if __name__ == '__main__':
    random.seed(world.SEED)
    np.random.seed(world.SEED)
    f_interation =world.DATA_PATH+'/app/Apps_for_Android_5.json'
    f_meta = world.DATA_PATH+'/app/meta_Apps_for_Android.json'
    f_out = world.DATA_PATH+'/app/app@20211208.pkl'
    test_ratio = 0.2

    all_data, user_src2tgt_dict, item_src2tgt_dict = load_interaction(f_interation)  # 加载数据，得到原id->正式id的dict
    user_size = len(user_src2tgt_dict)
    item_size = len(item_src2tgt_dict)

    cate_src2tgt_dict, item_cate_dict = load_meta(f_meta, item_src2tgt_dict)
    cate_size = len(cate_src2tgt_dict)
    print('user_num, item_num, cate_num', len(user_src2tgt_dict), len(item_src2tgt_dict), len(cate_src2tgt_dict))  # 3898, 11797, 29

    total_user_list = create_user_list(all_data, user_size) #把[u,i,r]转为list[u]={item...}
    print('avg. items per user: ', np.mean([len(u) for u in total_user_list]))  #11.68


    # ----------------------------------partition
    train_user_list, test_user_dict ,valid_user_dict= split_train_test(total_user_list,  # 划分train/test
                                                           test_size=test_ratio,time_order=False)
    print('min len(user_itemlist): ', min([len(ilist) for ilist in train_user_list]))
    print('train avg. items per user: ', np.mean([len(u) for u in train_user_list]))  # 165.57
    print('test avg. items per user: ', np.mean([len(test_user_dict[u]) for u in test_user_dict.keys()]))
    print('valid avg. items per user: ', np.mean([len(valid_user_dict[u]) for u in valid_user_dict.keys()]))

    dataset = {'user_size': user_size, 'item_size': item_size,
                   'train_user_list': train_user_list, 'test_user_dict': test_user_dict,
                   'valid_user_dict': valid_user_dict}
    with open(f_out, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


#[3]

