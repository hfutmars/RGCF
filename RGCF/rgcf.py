import tensorflow as tf
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utility.helper import *
from utility.batch_test import *

class RGCF(object):

    def __init__(self, data_config):
        # argument settings
        self.model_type = 'rgcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)

        self.n_layers = data_config['n_layers']

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.decay = data_config['decay']

        self.verbose = args.verbose

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        # initialization of model parameters
        self.weights = self._init_weights()

        self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        # self.u_bias = tf.nn.embedding_lookup(self.weights['u_bias'], self.users)
        # self.pos_i_bias = tf.nn.embedding_lookup(self.weights['i_bias'], self.pos_items)
        # self.neg_i_bias = tf.nn.embedding_lookup(self.weights['i_bias'], self.neg_items)

        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,transpose_b=True)
        # + tf.matmul(self.weights['c'], self.weights['i_bias'], transpose_a=False, transpose_b=True)

        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')

        # all_weights['u_bias'] = tf.Variable(tf.zeros([self.n_users, 1]), name='u_bias')
        # all_weights['i_bias'] = tf.Variable(tf.zeros([self.n_items, 1]), name='i_bias')
        # all_weights['c'] = tf.ones([args.batch_size*2, 1])

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat


    def _create_ngcf_embed(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        for k in range(0, self.n_layers):

            temp_embed = []

            for f in range(self.n_fold):

                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

        u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) \
                     # + self.u_bias + self.pos_i_bias
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) \
                     # + self.u_bias + self.neg_i_bias

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / args.batch_size
        # regularizer_b = tf.nn.l2_loss(self.u_bias) + tf.nn.l2_loss(self.pos_i_bias) + tf.nn.l2_loss(self.neg_i_bias)
        # regularizer_b = regularizer_b / args.batch_size

        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # mf_loss = tf.negative(tf.reduce_mean(maxi))

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer \
                   # + self.decay * regularizer_b

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)



if __name__ == '__main__':
    print('lr-->', args.lr)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    adj_type = 'norm_1'

    for decay in [1e-4]:
        print('decay-->', decay)
        for n_layers in [4]:
            print('layer-->', n_layers)
            data_generator.print_statistics()
            config = dict()
            config['n_users'] = data_generator.n_users
            config['n_items'] = data_generator.n_items
            config['decay'] = decay
            config['n_layers'] = n_layers

            norm_1 = data_generator.get_adj_mat()

            config['norm_adj'] = norm_1

            t0 = time()

            model = RGCF(data_config=config)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # sess = tf.Session()

            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')


            """
            *********************************************************
            Train.
            """
            loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
            stopping_step = 0
            should_stop = False

            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
                n_batch = data_generator.n_train // args.batch_size + 1

                for idx in range(n_batch):
                    users, pos_items, neg_items = data_generator.sample()
                    _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
                        [model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                        feed_dict={model.users: users, model.pos_items: pos_items,
                                   model.neg_items: neg_items})
                    loss += batch_loss
                    mf_loss += batch_mf_loss
                    emb_loss += batch_emb_loss
                    reg_loss += batch_reg_loss


                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                if (epoch + 1) % 10 != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                            epoch, time() - t1, loss, mf_loss, emb_loss)
                        print(perf_str)
                    continue

                t2 = time()
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)

                t3 = time()

                loss_loger.append(loss)
                rec_loger.append(ret['recall'])
                pre_loger.append(ret['precision'])
                ndcg_loger.append(ret['ndcg'])
                hit_loger.append(ret['hit_ratio'])

                if args.verbose > 0:
                    perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                               'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                               (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                                ret['recall'][0],
                                ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                    print(perf_str)

                cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                            stopping_step, expected_order='acc',
                                                                            flag_step=5)

                if should_stop == True:
                    break

            recs = np.array(rec_loger)
            pres = np.array(pre_loger)
            ndcgs = np.array(ndcg_loger)
            hit = np.array(hit_loger)

            best_rec_0 = max(recs[:, 0])
            idx = list(recs[:, 0]).index(best_rec_0)

            final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                          '\t'.join(['%.5f' % r for r in pres[idx]]),
                          '\t'.join(['%.5f' % r for r in hit[idx]]),
                          '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
            print(final_perf)

            save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
            ensureDir(save_path)
            f = open(save_path, 'a')

            f.write(
                'embed_size=%d, lr=%.5f, regs=%s, <<adj_type>>=%s\n\t%s\n'
                % (args.embed_size, args.lr, decay, adj_type, final_perf))
            f.close()
