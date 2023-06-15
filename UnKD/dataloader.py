"""
@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 000
"""
import os
import pickle
import sys

import torch
import world
import numpy as np

from world import cprint
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def validDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getAllCate(self):
        raise NotImplementedError

    def getOneCate(self,item):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    def set_item_cate_pair(self,pair):
        raise NotImplementedError

    def get_item_cate_pair(self):
        raise NotImplementedError

    def getOneUserPosItems(self, user):
        pass


# ----------------------------------------------------------------------------
class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    def __init__(self, config=world.config, path='../data/ml-1m/ml1m.pkl'):

        super().__init__()
        f_data = path
        with open(f_data, 'rb') as f:
            dataset = pickle.load(f)
            user_num, item_num = dataset['user_size'], dataset['item_size']
            # item_cate_dict = dataset['item_cate_dict']
            # item_cate_dict_s = dataset['item_cate_dict_s']
            train_user_list, test_user_dict,valid_user_dict= dataset['train_user_list'], dataset['test_user_dict'],dataset['valid_user_dict']

        self.train_user_list=train_user_list

        print('Load complete')
        # train or test
        cprint(f'loading [{path}]')
        self.__n_users =user_num
        self.__m_items = item_num
        #self.__k_cates = cate_num
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.__trainsize = 0
        self.validDataSize = 0
        self.testDataSize = 0

        for uid, items in enumerate(train_user_list):
            itemlen=len(items)
            trainUser.extend([uid] * itemlen)
            trainItem.extend(items)
            self.__trainsize += len(items)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        np.random.seed(world.SEED)
        for uid  in  test_user_dict.keys():
            test_items=test_user_dict[uid]
            itemlen = len(test_items)
            testUser.extend([uid] * itemlen)
            testItem.extend(test_items)
            self.testDataSize += len(test_items)
        for uid in valid_user_dict.keys():
            valid_items=valid_user_dict[uid]
            itemlen = len(valid_items)
            validUser.extend([uid] * itemlen)
            validItem.extend(valid_items)
            self.validDataSize += len(valid_items)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)


        self.Graph = None
        print(f"({self.n_users} X {self.m_items})")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.validDataSize} interactions for vailding")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.validDataSize + self.testDataSize) / self.n_users / self.m_items}"
        )

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.__n_users, self.__m_items),
            dtype='int')
        # pre-calculate
        self.__allPos = self.getUserPosItems(list(range(self.__n_users)))
        self.__testDict = self.build_dict(self.testUser, self.testItem)
        self.__validDict = self.build_dict(self.validUser, self.validItem)
        self.__trainDict = self.build_dict(self.trainUser, self.trainItem)
        print(f"{world.dataset} is ready to go")
        self.item_group,self.group_ratio,self.user_group_ratio,self.user_group_items=self.splitGroup(trainUser,trainItem,self.n_users,self.m_items,world.config['split_group'])
        # self.popular_group, self.unpopular_group,self.user_popular_group, self.user_unpopular_group = self.splitPopualrity(
        #     trainUser, trainItem, self.n_users, self.m_items)
        self.popular_group=self.item_group[0]
        self.unpopular_group=self.item_group[1]
        # self.popular_group=[]
        # for i in range(len(self.item_group)//2):
        #     print(i)
        #     self.popular_group+=self.item_group[i]
        # print('#############')
        # self.unpopular_group=[]
        # for i in range(len(self.item_group)//2,len(self.item_group),1):
        #     print(i)
        #     self.unpopular_group+=self.item_group[i]
        # self.item_group=[]
        # self.item_group.append(self.popular_group)
        # self.item_group.append(self.unpopular_group)
        self.count_weight()

    @property
    def n_users(self):
        return self.__n_users

    @property
    def m_items(self):
        return self.__m_items

    # @property
    # def k_cates(self):
    #     return self.__k_cates

    @property
    def trainDataSize(self):
        return self.__trainsize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def trainDict(self):
        return self.__trainDict

    @property
    def allPos(self):
        return self.__allPos


    def getOneUserPosItems(self, user):
       return self.train_user_list[user]

    @property
    def validDict(self):
        return self.__validDict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def build_dict(self, users, items):
        data = {}
        for i, item in enumerate(items):
            user = users[i]
            if data.get(user):
                data[user].append(item)
            else:
                data[user] = [item]
        return data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users,
                                         items]).astype('uint8').reshape(
                                             (-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.train_user_list[user])
        return posItems

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data,
                                                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero(as_tuple =False)
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items]))
            self.Graph = self.Graph.coalesce().cuda()
        return self.Graph



    def splitGroup(self,train_users,train_items,user_num,item_num,group_num=3):
        print('***begin group***')
        item_count={}
        listLen=len(train_items)
        count_sum = 0
        for i in range(listLen):
                k=train_items[i]
                if k not in item_count:
                    item_count[k]=0
                item_count[k]+=1
                count_sum += 1
        count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        group_aver = count_sum / group_num
        item_group=[]
        last_num=0
        temp_group = []
        temp_count = 0
        for l in range(len(count)):
            i, j = count[l][0], count[l][1]
            temp_group.append(i)
            temp_count+=j
            if temp_count>group_aver:
                if len(temp_group) == 1:
                    item_group.append(temp_group)
                    temp_group = []
                    temp_count = 0
                    continue
                temp_group.remove(i)
                item_group.append(temp_group)
                temp_group = []
                temp_group.append(i)
                temp_count=j
        if len(temp_group) > 0:
            if temp_count>group_aver/2:
                item_group.append(temp_group)
            else:
                if len(item_group)>0:
                    item_group[-1].extend(temp_group)
                else:
                    item_group.append(temp_group)

        print('group_len')
        for i in range(len(item_group)):
            print(len(item_group[i]))
        cate_ratio=[]
        temp = 0
        print('popualrity sum')
        for i in range(len(item_group)):
            tot = 0
            tot_n = 0
            for j in range(len(item_group[i])):
                tot += item_count[item_group[i][j]]
                tot_n += 1
            print(tot)
            cate_ratio.append(tot / tot_n)
        print(cate_ratio)
        maxP = max(cate_ratio)
        minP = min(cate_ratio)
        for i in range(len(cate_ratio)):
            cate_ratio[i] = (maxP + minP) - cate_ratio[i]
            temp += cate_ratio[i]
        for i in range(len(cate_ratio)):
            cate_ratio[i] = round((cate_ratio[i] / temp), 2)
        # cate_ratio.reverse()
        for i in range(len(cate_ratio)):
            if cate_ratio[i] < 0.1:
                cate_ratio[i] = 0.1
        print(cate_ratio)

        user_group_ratio=[[] for j in range(user_num)]
        user_group_items = [[] for j in range(user_num)]
        for i in range(user_num):
            user_group_items[i] = [[] for j in range(group_num)]
            user_group_ratio[i] = [0 for j in range(group_num)]
        for i in range(len(train_items)):
            for k in range(len(item_group)):
                if train_items[i] in item_group[k]:
                    user_group_ratio[train_users[i]][k]+=1
                    user_group_items[train_users[i]][k].append(train_items[i])
        # for i in range(len(user_group_ratio)):
        #     for j in range(len(user_group_ratio[i])):
        #         tot=sum(user_group_ratio[i])
        #         user_group_ratio[i][j] = round(user_group_ratio[i][j] / tot, 2)
        print('***end group***')
        return item_group, cate_ratio, user_group_ratio,user_group_items

    def splitPopualrity(self, train_users, train_items, user_num, item_num):
        print('***begin popularity***')
        item_group, _, _, user_group_items = self.splitGroup(
            train_users, train_items, user_num, item_num, 2)
        popualr_group=item_group[0]
        unpopular_group=item_group[1]
        user_popualr_items = [[] for j in range(user_num)]
        user_unpopualr_items = [[] for j in range(user_num)]
        for i in range(len(user_group_items)):
            user_popualr_items[i]=user_group_items[i][0]
            user_unpopualr_items[i] = user_group_items[i][1]

        return popualr_group, unpopular_group, user_popualr_items, user_unpopualr_items

    def count_weight(self):
        cate_count=[1 for i in range(self.m_items)]
        for i in range(len(self.trainItem)):
            cate_count[self.trainItem[i]]+=1

        #ips
        if world.model_name == 'IPSModel':
            print('IPSModel')
            P_Oeq1=1/(self.m_items)
            item_count=torch.Tensor(cate_count)
            P_y_givenO=item_count/(self.n_users*self.m_items)
            max_v=max(item_count)
            min_V=min(item_count)
            P_y=item_count/(max_v+1)
            Propensity = P_y
            print(max(Propensity))
            print(min(Propensity))
            #self.item_weight=torch.Tensor(torch.pow(Propensity,world.config['alpha']))
            self.item_weight =Propensity
        # pda pcc
        else:
            item_weight = torch.Tensor(cate_count)
            sum_weight = sum(item_weight)
            self.item_weight = torch.Tensor(np.power(item_weight/sum_weight,world.config['alpha']))
        print(self.item_weight)
# ----------------------------------------------------------------------------
# this dataset is for debugging
