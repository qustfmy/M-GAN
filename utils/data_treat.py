from abc import ABC
import numpy as np
import torch
from torch.utils import data
from data_utils import read_data, read_data2, split_data, build_vocab, build_dict, pad_data, star_embdding, read_file
import os
import pickle


def get_project_path():
    """得到项目路径"""
    project_path = os.path.join(
        os.path.dirname(__file__),
    )
    return project_path


BASE_DIR = get_project_path()
# pick_path = os.path.join(BASE_DIR, 'pkl/cossim/pad_data.pkl')
# read_train_data = os.path.join(BASE_DIR, 'pkl/cossim/read_train_data.pkl')
# read_val_data = os.path.join(BASE_DIR, 'pkl/cossim/read_val_data.pkl')
# read_eval_data = os.path.join(BASE_DIR, 'pkl/cossim/read_eval_data.pkl')
# embed_path = os.path.join(BASE_DIR, 'pkl/cossim/embed_data.pkl')
# pick_path = os.path.join(BASE_DIR,'pkl/pad_data.pkl')
# read_train_data =os.path.join(BASE_DIR,'pkl/read_train_data.pkl')
# read_val_data = os.path.join(BASE_DIR,'pkl/read_val_data.pkl')
# read_eval_data = os.path.join(BASE_DIR,'pkl/read_eval_data.pkl')
# embed_path = os.path.join(BASE_DIR,'pkl/embed_data.pkl')\
pick_path = 'pkl/superuser/pad_data.pkl'
read_train_data = 'pkl/superuser/read_train_data.pkl'
read_val_data = 'pkl/superuser/read_val_data.pkl'
read_eval_data = 'pkl/superuser/read_eval_data.pkl'
embed_path = 'pkl/superuser/embed_data.pkl'


class config_data(data.Dataset, ABC):
    def __init__(self, data_path_train, data_path_val, data_path_eval, USE_CUDA):

        if not os.path.exists(read_train_data):  # 如果数据模型文件不存在
            print('------------')
            print('read train data')
            # 返回X：QA对 Y：是标签 Z是qk和ak
            X_train, Y_train, Z_train, qat_best, qt_best = read_data2(
                data_path_train)  # 传入训练数据集
            with open(read_train_data, 'wb') as read_file:
                print("开始转存")
                pickle.dump([X_train, Y_train, Z_train,
                            qat_best, qt_best], read_file)  # 形成
        else:
            with open(read_train_data, 'rb') as read_file:
                X_train, Y_train, Z_train, qat_best, qt_best = pickle.load(
                    read_file)

        if not os.path.exists(read_val_data):
            print('------------')
            print('read val data')
            X_val, Y_val, Z_val, qav_best, qv_best = read_data2(data_path_val)
            with open(read_val_data, 'wb') as read_file:
                pickle.dump(
                    [X_val, Y_val, Z_val, qav_best, qv_best], read_file)
        else:
            with open(read_val_data, 'rb') as read_file:
                X_val, Y_val, Z_val, qav_best, qv_best = pickle.load(read_file)

        if not os.path.exists(read_eval_data):
            print('------------')
            print('read eval data')
            X_eval, Y_eval, Z_eval, qae_best, qe_best = read_data2(
                data_path_eval)
            with open(read_eval_data, 'wb') as read_file:
                pickle.dump([X_eval, Y_eval, Z_eval,
                            qae_best, qe_best], read_file)
        else:
            with open(read_eval_data, 'rb') as read_file:
                X_eval, Y_eval, Z_eval, qae_best, qe_best = pickle.load(
                    read_file)
        # xq xa qk ak
        self.Xqtrain, self.Xatrain, self.Qktrain, self.Aktrain = split_data(
            X_train, Z_train)  # 传入QA对和qk、ak
        self.Xqval, self.Xaval, self.Qkval, self.Akval = split_data(
            X_val, Z_val)
        self.Xqeval, self.Xaeval, self.Qkeval, self.Akeval = split_data(
            X_eval, Z_eval)

        self.question_maxlen = max([len(qapair[0].split())
                                   for qapair in X_val])  # 得到问题的最大长度
        self.answer_maxlen = max([len(qapair[1].split())
                                 for qapair in X_val])  # 得到答案的最大长度
        vocab = build_vocab(self.Xqtrain, self.Xatrain,
                            self.Xqval, self.Xaval)  # 建立词表，把所有数据按照句子切分单词append
        self.word2index, self.index2word, self.index2target, self.vocab_size = build_dict(
            Y_train, vocab)  # 传入标签和vocab

        if not os.path.exists(pick_path):
            with open(pick_path, 'wb') as pad_file:
                print('------------')
                print('pad data')
                Xq_train, Xa_train, Qk_train, Ak_train = pad_data(self.Xqtrain, self.Xatrain, self.Qktrain,
                                                                  self.Aktrain, self.word2index)  # {'pad': 0, 'unk': 1,'hello':2,}
                Xq_val, Xa_val, Qk_val, Ak_val = pad_data(self.Xqval, self.Xaval, self.Qkval, self.Akval,
                                                          self.word2index)
                Xq_eval, Xa_eval, Qk_eval, Ak_eval = pad_data(self.Xqeval, self.Xaeval, self.Qkeval, self.Akeval,
                                                              self.word2index)
                pickle.dump(
                    [Xq_train, Xa_train, Qk_train, Ak_train, Xq_val, Xa_val, Qk_val, Ak_val, Xq_eval, Xa_eval, Qk_eval,
                     Ak_eval], pad_file)
        else:
            with open(pick_path, 'rb') as pad_file:
                Xq_train, Xa_train, Qk_train, Ak_train, Xq_val, Xa_val, Qk_val, Ak_val, Xq_eval, Xa_eval, Qk_eval, Ak_eval = pickle.load(
                    pad_file)

        self.qa_best = qat_best + qav_best
        self.q_best = qt_best + qav_best
        self.vocab_size = len(self.word2index)
        self.embedding_size = 300
        self.pretrained = []
        self.zero_embeddings = np.zeros((self.embedding_size,))
        self.count = 0

        if not os.path.exists(embed_path):
            with open(embed_path, 'wb') as embed_file:
                print('read embedding')
                # starspace_embeddings = star_embdding('./subdata/super-user/vectors.txt')
                starspace_embeddings = star_embdding('./glove.840B.300d.txt')
                for key in self.word2index.keys():
                    if key in starspace_embeddings:
                        self.count += 1
                        self.pretrained.append(starspace_embeddings[key])
                    else:
                        self.pretrained.append(np.random.randn(300))
                self.pretrained_vectors = np.vstack(self.pretrained)
                pickle.dump(self.pretrained_vectors, embed_file)
        else:
            with open(embed_path, 'rb') as embed_file:
                self.pretrained_vectors = pickle.load(embed_file)

        self.train_question_inps = torch.tensor(Xq_train)
        self.train_answer_inps = torch.tensor(Xa_train)
        self.train_tgts = torch.tensor(Y_train)
        self.train_Qk_inps = torch.tensor(Qk_train)
        self.train_Ak_inps = torch.tensor(Ak_train)
        self.val_question_inps = torch.tensor(Xq_val)
        self.val_answer_inps = torch.tensor(Xa_val)
        self.val_tgts = torch.tensor(Y_val)
        self.val_Qk_inps = torch.tensor(Qk_val)
        self.val_Ak_inps = torch.tensor(Ak_val)
        self.eval_question_inps = torch.tensor(Xq_eval)
        self.eval_answer_inps = torch.tensor(Xa_eval)
        self.eval_tgts = torch.tensor(Y_eval)
        self.eval_Qk_inps = torch.tensor(Qk_eval)
        self.eval_Ak_inps = torch.tensor(Ak_eval)
        if USE_CUDA:
            self.train_question_inps = self.train_question_inps.cuda()
            self.train_answer_inps = self.train_answer_inps.cuda()
            self.train_tgts = self.train_tgts.cuda()
            self.train_Qk_inps = self.train_Qk_inps.cuda()
            self.train_Ak_inps = self.train_Ak_inps.cuda()
            self.val_question_inps = self.val_question_inps.cuda()
            self.val_answer_inps = self.val_answer_inps.cuda()
            self.val_tgts = self.val_tgts.cuda()
            self.val_Qk_inps = self.val_Qk_inps.cuda()
            self.val_Ak_inps = self.val_Ak_inps.cuda()
            # self.eval_question_inps = self.eval_question_inps.cuda()
            # self.eval_answer_inps = self.eval_answer_inps.cuda()
            # self.eval_tgts = self.eval_tgts.cuda()
            # self.eval_Qk_inps = self.eval_Qk_inps.cuda()
            # self.eval_Ak_inps = self.eval_Ak_inps.cuda()
