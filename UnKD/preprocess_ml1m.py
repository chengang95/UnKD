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


def load_interaction(data_dir):
    all_data = []
    user_src2tgt_dict, item_src2tgt_dict = {}, {}
    user_count={}
    f = open(data_dir, 'r')
    for eachline in f:
        # eachline = json.loads(eachline)
        # u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        eachline = eachline.split('::')
        u, i, r, t = int(eachline[0]), int(eachline[1]), float(eachline[2]), int(eachline[3])
        if r > 0:
            if u not in user_count:
                user_count[u] =0
            user_count[u]+=1
    f.close()


    f = open(data_dir, 'r')
    for eachline in f:
        #eachline = json.loads(eachline)
        #u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        eachline = eachline.split('::')
        u, i, r, t = int(eachline[0]), int(eachline[1]), float(eachline[2]), int(eachline[3])
        if r>0 and user_count[u]>=20:
            if u not in user_src2tgt_dict:
                user_src2tgt_dict[u] = len(user_src2tgt_dict)
            user = user_src2tgt_dict[u]
            if i not in item_src2tgt_dict:
                item_src2tgt_dict[i] = len(item_src2tgt_dict)
            item = item_src2tgt_dict[i]
            all_data.append([user, item, t])
    return all_data, user_src2tgt_dict, item_src2tgt_dict

def create_user_list(all_data, user_size):
    user_list = [dict() for u in range(user_size)]
    count = 0
    for u, i, t in all_data:
        user_list[u][i] = t
    return user_list

def deal_train_list(train_user_list,user_list,user_size,item_cate_dict_s,current_cate_list):
    t_user_list = [list() for u in range(user_size)]
    count = 0
    current_cate_list=set(current_cate_list)
    for u in range(len(train_user_list)):
        for i in range(len(train_user_list[u])):
            c=set(item_cate_dict_s[train_user_list[u][i]])
            if len(c&current_cate_list)>0:
                t_user_list[u].append(train_user_list[u][i])
                count += 1
            elif user_list[u][train_user_list[u][i]]>0:
                t_user_list[u].append(train_user_list[u][i])
                count +=1
    print('intreaction num :'+str(count))
    return t_user_list




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

    sampel_user=np.random.choice(range(len(user_list)), size=int(len(user_list)*0.5), replace=False)
    test_user_dict={}
    valid_user_dict={}
    for u in range(len(test_user_list)):
        if u in sampel_user:
            test_user_dict[u]=test_user_list[u]
        else:
            valid_user_dict[u]=test_user_list[u]
    print("the number of user of no test item   %d" % count1)
    return train_user_list, test_user_dict,valid_user_dict





if __name__ == '__main__':
    random.seed(world.SEED)
    np.random.seed(world.SEED)
    test_ratio = 0.2

    f_userinfo = world.DATA_PATH+'/ml1m/users.dat'
    f_iteminfo = world.DATA_PATH+'/ml1m/movies.dat'
    f_dataall = world.DATA_PATH+'/ml1m/ratings.dat'
    f_out = world.DATA_PATH+'/ml1m/ml1m@20230212'+'.pkl'

    #load interaction
    all_data, user_src2tgt_dict, item_src2tgt_dict = load_interaction(f_dataall)
    user_size = len(user_src2tgt_dict)
    item_size = len(item_src2tgt_dict)

    # ------------------------------------------------load item
    cate_list = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    cate_src2tgt_dict = {f: i for i, f in enumerate(cate_list)}
    print(cate_src2tgt_dict)
    all_cate_list=list(cate_src2tgt_dict.values())
    light_cate_list =[]
    current_cate_list=[]
    for c in light_cate_list:
        if c in cate_src2tgt_dict.keys():
            current_cate_list.append(cate_src2tgt_dict[c])

    genre_count = {}
    item_cate_dict = {}
    item_cate_dict_s={}
    movie_df = pd.read_csv(f_iteminfo, sep=r'::', header=None, engine='python')
    for _, row in movie_df.iterrows():
        row_tmp = row.tolist()
        i_id = row_tmp[0]
        if i_id not in item_src2tgt_dict:   #item出现在item_cate列表中，但不在u-i-r列表中，应该去掉
            continue
        cate_list = row_tmp[2].strip().split('|')
        cate_list_tmp=[]
        for c in cate_list:
            if c in cate_src2tgt_dict.keys():
                cate_list_tmp.append(cate_src2tgt_dict[c])
        cate_list=cate_list_tmp
        # item_feature_dict[i_id] = cate_list
        cate = np.random.choice(cate_list)
        #cate=cate_list[0]
        cate_list_sigle=[]
        cate_list_sigle.append(cate)
        #genre_count[cate].append(i_id)
        for cate in cate_list:
            if cate not in genre_count:
                genre_count[cate]=0
            genre_count[cate]+=1

        #item_cate_dict[item_src2tgt_dict[i_id]] = cate_list_sigle
        item_cate_dict_s[item_src2tgt_dict[i_id]]=cate_list
    # item_cate_dict = sorted(item_cate_dict.items(), key=lambda mydict: mydict[0], reverse=False)
    # item_cate_dict = {i:j for i,j in item_cate_dict}
    genre_count = sorted(genre_count.items(), key=lambda mydict: mydict[0], reverse=False)
    genre_count = {i:j for i,j in genre_count}
    item_cate_dict_s = sorted(item_cate_dict_s.items(), key=lambda mydict: mydict[0], reverse=False)
    item_cate_dict_s = {i: j for i, j in item_cate_dict_s}
    for i in range(len(item_cate_dict_s)):
        for j in genre_count.keys():
            if j in item_cate_dict_s[i]:
                item_cate_dict[i]=[j]
                break
    cate_size = len(genre_count)
    # for i in genre_count.keys():
    #     print(str(i)+":"+str(len(genre_count[i])))

    total_user_list = create_user_list(all_data, user_size)  # 把[u,i,r]转为list[u]={item...}

    user_num=len(total_user_list)
    print('user_size, item_size,inter_num:', user_size, item_size, len(all_data))

    # ----------------------------------partition
    train_user_list, test_user_dict,valid_user_dict= split_train_test(total_user_list,  # 划分train/test
                                                           test_size=test_ratio,time_order=False)
    #train_user_list=deal_train_list(train_user_list,total_user_list,user_size,item_cate_dict_s,current_cate_list)
    print('min len(user_itemlist): ', min([len(ilist) for ilist in train_user_list]))
    print('train avg. items per user: ', np.mean([len(u) for u in train_user_list]))  # 165.57
    print('test avg. items per user: ', np.mean([len(test_user_dict[u]) for u in test_user_dict.keys()]))
    print('valid avg. items per user: ', np.mean([len(valid_user_dict[u]) for u in valid_user_dict.keys()]))
    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_dict': test_user_dict,'valid_user_dict':valid_user_dict,}
    with open(f_out, 'wb') as f:
        #json.dump(dataset, f)
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

#[4,7,0]