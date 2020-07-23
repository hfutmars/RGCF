import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import collections

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        # self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        self.coo_R = self.R.tocoo()

        self.all_head_list, self.all_tail_list, self.all_v_list = self._get_all_entity_list_()

    def get_adj_mat(self):
        try:
            t1 = time()

            norm_adj_mat_1 = sp.load_npz(self.path + '/s_norm_adj_mat_1.npz')

            print('already load adj matrix', norm_adj_mat_1.shape, time() - t1)

        except Exception:
            norm_adj_mat_1 = self.create_adj_mat()

            sp.save_npz(self.path + '/s_norm_adj_mat_1.npz', norm_adj_mat_1)

        return norm_adj_mat_1

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_symetric(adj):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            d_inv_sqrt_last = np.power(rowsum, -0.4).flatten()
            d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
            d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

            return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()

        norm_adj_mat_1 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]))
        # norm_adj_mat_1 = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat_1.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def _get_all_entity_list_(self):

        all_head_list = list(self.coo_R.row)
        all_tail_list = list(self.coo_R.col)
        all_v_list = list(self.coo_R.data)

        org_h_dict = dict()
        for idx, h in enumerate(all_head_list):
            if h not in org_h_dict:
                org_h_dict[h] = [[], []]
            org_h_dict[h][0].append(all_tail_list[idx])
            org_h_dict[h][1].append(all_v_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = np.array(org_t_list)[sort_order]
            sort_v_list = np.array(org_v_list)[sort_order]

            sorted_h_dict[h] = [sort_t_list, sort_v_list]

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_v_list = [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_v_list += list(vals[1])

        return new_h_list, new_t_list, new_v_list



