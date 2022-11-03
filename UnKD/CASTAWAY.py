# ----------------------------------------------------------------------------
# from Procedure
def BPR_train_DNS_batch(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    global item_count
    Recmodel: PairWiseModel = recommend_model
    Recmodel.train()
    if item_count is None:
        item_count = torch.zeros(dataset.m_items)
    bpr: utils.BPRLoss = loss_class
    # S,sam_time = UniformSample_DNS_deter(allusers, dataset, world.DNS_K)
    S, negItems,NEG_scores, sam_time = UniformSample_DNS_batch(dataset, Recmodel, world.DNS_K)

    print(f"DNS[pre-sample][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = S[:, 0].long()
    posItems = S[:, 1].long()
    negItems = negItems.long()
    negScores = NEG_scores.float()
    print(negItems.shape, negScores.shape)
    users, posItems, negItems, negScores = utils.TO(users, posItems, negItems, negScores)
    users, posItems, negItems, negScores = utils.shuffle(users, posItems, negItems, negScores)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    DNS_time = time()
    DNS_time1 = 0.
    DNS_time2 = 0.
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          batch_scores)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   negScores,
                                                   batch_size=world.config['bpr_batch_size'])):
        # batch_neg, sam_time = DNS_sampling_neg(batch_users, batch_neg, dataset, Recmodel)
        batch_neg, sam_time = DNS_sampling_batch(batch_neg, batch_scores)
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        item_count[batch_neg] += 1
        DNS_time1 += sam_time[0]
        DNS_time2 += sam_time[2]
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    print(f"DNS[sampling][{time()-DNS_time:.1f}={DNS_time1:.2f}+{DNS_time2:.2f}]")
    np.savetxt(os.path.join(world.CODE_PATH, f"counts/count_{world.dataset}_{world.DNS_K}.txt"),item_count.numpy())
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"


"""
def UniformSample_DNS_batch(self, epoch, batch_score_size=512):
        # with torch.no_grad():
        #     if epoch >= self.start_epoch:
        #         self.start = True
        #     total_start = time()
        #     dataset = self.dataset
        #     dns_k = self.dns_k
        #     user_num = dataset.trainDataSize
        #     per_user_num = user_num // dataset.n_users + 1
        #     allPos = dataset.allPos
        #     S = []
        #     NEG_scores = []
        #     NEG_scores_teacher = []
        #     sample_time1 = 0.
        #     sample_time2 = 0.
        #     sample_time3 = 0.
        #     sample_time4 = 0.
        #     BinForUser = np.zeros(shape = (dataset.m_items, )).astype("int")
        #     # sample_shape = int(dns_k*1.5)+1
        #     BATCH_SCORE = None
        #     BATCH_SCORE_teacher = None
        #     now = 0
        #     NEG = np.zeros((per_user_num*dataset.n_users, dns_k))
        #     STUDENT = torch.zeros((per_user_num*dataset.n_users, dns_k))
        #     TEACHER = torch.zeros((per_user_num*dataset.n_users, dns_k))
        #     for user in range(dataset.n_users):
        #         start1 = time()
        #         if now >= batch_score_size:
        #             del BATCH_SCORE
        #             BATCH_SCORE = None
        #             BATCH_SCORE_teacher = None
        #         if BATCH_SCORE is None:
        #             left_limit = user+batch_score_size
        #             batch_list = torch.arange(user, left_limit) if left_limit <= dataset.n_users else torch.arange(user, dataset.n_users)
        #             BATCH_SCORE = self.student.getUsersRating(batch_list).cpu()
        #             BATCH_SCORE_teacher = self.teacher.getUsersRating(batch_list, t1=self.t1, t2=self.t2)
        #             now = 0
        #         sample_time1 += time()-start1

        #         start2 = time()
        #         scoreForuser = BATCH_SCORE[now]
        #         scoreForuser_teacher = BATCH_SCORE_teacher[now]
        #         # scoreForuser_teacher = BATCH_SCORE[now]
        #         now += 1
        #         posForUser = allPos[user]
        #         if len(posForUser) == 0:
        #             continue
        #         BinForUser[:] = 0
        #         BinForUser[posForUser] = 1
        #         NEGforUser = np.where(BinForUser == 0)[0]
        #         for i in range(per_user_num):
        #             start3 = time()
        #             posindex = np.random.randint(0, len(posForUser))
        #             positem = posForUser[posindex]
        #             negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
        #             negitems = NEGforUser[negindex]
        #             add_pair = (user, positem)
        #             # NEG_scores.append(scoreForuser[negitems])
        #             STUDENT[user*per_user_num + i, :] = scoreForuser[negitems]
        #             TEACHER[user*per_user_num + i, :] = scoreForuser_teacher[negitems]
        #             # NEG_scores_teacher.append(scoreForuser_teacher[negitems])

        #             sample_time3 += time()-start3
        #             start4 = time()
        #             S.append(add_pair)
        #             NEG[user*per_user_num + i, :] = negitems
        #             sample_time4 += time() - start4
        #         sample_time2 += time() - start2
        # # ===========================
        # if self.start:
        #     self.W *= self.beta
        # return torch.Tensor(S), torch.from_numpy(NEG), torch.stack(NEG_scores), torch.stack(NEG_scores_teacher),[time() - total_start, sample_time1, sample_time2, sample_time3, sample_time4]
        return torch.Tensor(S), torch.from_numpy(NEG), STUDENT, TEACHER,[time() - total_start, sample_time1, sample_time2, sample_time3, sample_time4]
"""

'''
# ============================================================================
# multi-core sampling, not yet
def UniformSample_DNS_neg_multi(users, dataset, dns_k):
    """
    the original impliment of BPR Sampling in LightGCN
    NOTE: we can sample a whole epoch data at one time
    :return:
        np.array
    """
    global ALLPOS
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    ALLPOS = dataset.allPos
    S = []
    negItems = []
    sample_time1 = 0.
    sample_time2 = 0.
    pool = multiprocessing.Pool(world.CORES)
    dns_ks = [dns_k]*user_num
    m_itemss = [dataset.m_items]*user_num
    X = zip(users, m_itemss, dns_ks)
    results = pool.map(UniformSample_user, X)
    results = [data for data in results if data is not None]
    S = np.vstack(results)
    total = time() - total_start
    return S, [total, sample_time1, sample_time2]

def UniformSample_user(X):
    user = X[0]
    m_items = X[1]
    dns_k = X[2]
    posForUser = ALLPOS[user]
    BinForUser = np.zeros(shape = (m_items, )).astype("int")
    if len(posForUser) == 0:
        return None
    BinForUser[posForUser] = 1
    start = time()
    posindex = np.random.randint(0, len(posForUser))
    positem = posForUser[posindex]
    negitems = []
    while True:
        negitems = np.random.randint(0, m_items, size=(dns_k, ))
        if np.sum(BinForUser[negitems]) > 0:
            continue
        else:
            break
    add_pair = [user, positem]
    add_pair.extend(negitems)
    return np.array(add_pair).astype('int')
'''

    # ----------------------------------------------------------------------------
    # ranking
    '''
    def ranking(self, batch_users, batch_pos, batch_neg):
        """
        with grad
        """
        STUDENT = self.student
        TEACHER = self.teacher
        dns_k = self.dns_k
        times = []
        
        with Timer(times):    
            NegItems = batch_neg
            negitem_vector = NegItems.reshape((-1, )) # dns_k * |users|
            user_vector = batch_users.repeat((dns_k, 1)).t().reshape((-1,))
            
            student_scores = STUDENT(user_vector, negitem_vector)
            student_scores = student_scores.reshape((-1, dns_k))
            
            teacher_scores = TEACHER(user_vector, negitem_vector)
            teacher_scores = teacher_scores.reshape((-1, dns_k))
            teacher_pos_scores = TEACHER(batch_users, batch_pos)
            
            _, top1 = student_scores.max(dim=1)
            idx = torch.arange(len(batch_users))
            negitems = NegItems[idx, top1]
            weights = self.sigmoid((teacher_pos_scores - teacher_scores[idx, top1])/self.t)
        
        all_pairs = self.pairs.T
        pairs = self.pairs.T
        rank_loss = torch.tensor(0.).to(world.DEVICE)
        total_err = 0
        
        with Timer(times):
            for i, user in enumerate(batch_users) :
                # pairs = all_pairs[:, np.random.randint(all_pairs.shape[1], size=(8, ))]
                teacher_rank = (teacher_scores[i][pairs[0]] > teacher_scores[i][pairs[1]])
                student_rank = (student_scores[i][pairs[0]] > student_scores[i][pairs[1]])
                err_rank = torch.logical_xor(teacher_rank, student_rank)
                total_err += torch.sum(err_rank)
                should_rank_g = torch.zeros_like(teacher_rank).bool()
                should_rank_l = torch.zeros_like(teacher_rank).bool()
                # use teacher to confirm wrong rank
                should_rank_g[err_rank] = teacher_rank[err_rank]
                should_rank_l[err_rank] = (~teacher_rank)[err_rank]
                if torch.any(should_rank_g):
                    rank_loss += torch.mean(softplus(
                        (student_scores[i][pairs[1]] - student_scores[i][pairs[0]])[should_rank_g]
                    ))# should rank it higher
                if torch.any(should_rank_l):
                    rank_loss += torch.mean(softplus(
                        (student_scores[i][pairs[0]] - student_scores[i][pairs[1]])[should_rank_l]
                    ))# should rank it lower
                if torch.isnan(rank_loss) or torch.isinf(rank_loss):
                    print("student", student_scores[i])
                    print("pos", (student_scores[i][pairs[1]] - student_scores[i][pairs[0]])[should_rank_g])
                    print("neg", (student_scores[i][pairs[0]] - student_scores[i][pairs[1]])[should_rank_l])
                    exit(0)
        rank_loss /= len(batch_users)
        rank_loss = rank_loss*self.beta
        print(f"{total_err.item()}-> ", end='')
        return negitems, weights, rank_loss, np.asanyarray(times)
    '''
    # ----------------------------------------------------------------------------
    # method 4
    # def weight_pair(self, batch_users, batch_pos, batch_neg, epoch):
    #     with torch.no_grad():
    #         start = time()
    #         student_score = userAndMatrix(batch_users, batch_neg, self.student)
    #         teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)
    #         batch_list = torch.arange(0, len(batch_neg))
    #         _, student_max = torch.max(student_score, dim=1)
    #         Items = batch_neg[batch_list, student_max]
    #         weights = self.teacher.pair_score(batch_users, batch_pos, Items)
    #         weights = weights
    #         return Items, weights, None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------
    # method 2
    # def random_indicator(self, batch_users, batch_pos, batch_neg, epoch):
    #     start = time()
    #     if self.start:
    #         student_score = userAndMatrix(batch_users, batch_neg, self.student)
    #         teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)

    #         batch_list = torch.arange(0, len(batch_neg))
    #         _, student_max = torch.max(student_score, dim=1)
    #         teacher_p = self.soft(-teacher_score/self.T)
    #         teacher_index = torch.multinomial(teacher_p, 1).squeeze()
    #         student_neg = batch_neg[batch_list, student_max]
    #         teacher_neg = batch_neg[batch_list, teacher_index]
    #         Items = torch.zeros((len(batch_neg), )).to(world.DEVICE).long()
    #         P_bern = torch.ones((len(batch_neg), ))*self.W
    #         indicator = torch.bernoulli(P_bern).bool()
    #         Items[indicator] = student_neg[indicator]
    #         Items[~indicator] = teacher_neg[~indicator]
    #         return Items, None,[time()-start, 0, 0]
    #     else:
    #         return self.DNS(batch_neg, student_score), None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------
    # method 3
    # def max_min(self, batch_users, batch_pos, batch_neg, epoch):
    #     start = time()
    #     if self.start:
    #         student_score = userAndMatrix(batch_users, batch_neg, self.student)
    #         teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)
    #         batch_list = torch.arange(0, len(batch_neg))
    #         _, student_max = torch.max(student_score, dim=1)
    #         _, teacher_min = torch.min(teacher_score, dim=1)
    #         student_neg = batch_neg[batch_list, student_max]
    #         teacher_neg = batch_neg[batch_list, teacher_min]
    #         Items = torch.zeros((len(batch_neg), )).to(world.DEVICE).long()
    #         P_bern = torch.ones((len(batch_neg), ))*self.W
    #         indicator = torch.bernoulli(P_bern).bool()
    #         Items[indicator] = student_neg[indicator]
    #         Items[~indicator] = teacher_neg[~indicator]
    #         return Items, None,[time()-start, 0, 0]
    #     else:
    #         return self.DNS(batch_neg, student_score), None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------