import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
# from keras.preprocessing.sequence import pad_sequences
# import data
import data_test
import torch.utils.data as data_util
from model import Encoder
import torch.nn.functional as F
import data_utils
from tqdm import tqdm
import os
import data_utils
import time
import pickle
import random
import matplotlib.pyplot as plt
from LabelSmoothingCrossEntropy import *
from sklearn.metrics import dcg_score


def get_project_path():
    """得到项目路径"""
    project_path = os.path.join(
        os.path.dirname(__file__),
    )
    return project_path


BASE_DIR = get_project_path()
# CUDA_VISIBLE_DEVICES=1 python train.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def flatten(l): return [item for sublist in l for item in sublist]


random.seed(1024)
torch.manual_seed(777)
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
batch_size = 64

print('deal data')

strain_data = data_test.config_data('subdata/Q2CQ/train_data_4cls8',
                                    'subdata/Q2CQ/validate_data_4cls40',
                                    'subdata/Q2CQ/evaluation_data_4cls40', USE_CUDA)
train_data = data_util.TensorDataset(strain_data.train_question_inps,  # Xq_train
                                     strain_data.train_answer_inps,  # Xa_train
                                     strain_data.train_tgts,  # Y_train
                                     strain_data.train_Qk_inps,  # qk
                                     strain_data.train_Ak_inps)  # ak
train_loader = data_util.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
print('-----------Finish')
print('CUDA使用：{}'.format(USE_CUDA))
print('deal evaluation data')
evaluation_qa_pairs = data_utils.eval_Data(
    strain_data.qa_best, strain_data.q_best, 'pkl/superuser2/evaluate_data.pkl')  # evaluation_qa_pairs


EPOCH = 15
KERNEL_SIZES = [3, 4, 5]
KERNEL_DIM = 300
LR = 0.001

#model_path = None
model_path = './ans_encoder.pt'
#model_path = './pkl/superuser2/ans_encoder.pt'
#model_path = None
rank_path = 'pkl/superuser2/rank_data.pkl'


# 词典大小
vocab_size = strain_data.vocab_size
pretrained_vectors = strain_data.pretrained_vectors

encoder = Encoder(USE_CUDA, pretrained_vectors, vocab_size,
                  300, 3, KERNEL_DIM, KERNEL_SIZES).to(device)
encoder.init_weights(pretrained_vectors)
if USE_CUDA:
    encoder = encoder.cuda()


#loss_function = nn.CrossEntropyLoss()
loss_function = LabelSmoothingCrossEntropy()
opt = optim.Adam(encoder.parameters(), lr=LR)
# opt = optim.Adam(encoder.parameters(), lr=LR, weight_decay=0.01)
encoder = encoder.train()
print(encoder)

if model_path is None:
    for epoch in range(EPOCH):
        start_time = time.time()
        avg_cost = 0  # 平均损失
        total_batch = len(train_data) // batch_size  # 表示整数除法，它可以返回商的整数部分

        for i, (batch_xqs, batch_xas, batch_ys, batch_Qk, batch_Ak) in enumerate(train_loader):
            Xq = Variable(batch_xqs)
            Xa = Variable(batch_xas)
            Y = Variable(batch_ys)
            Qk = Variable(batch_Qk)
            Ak = Variable(batch_Ak)
            encoder.zero_grad()
            preds = encoder(Xq, Xa, Qk, Ak)
            cost = loss_function(preds, Y)
            cost.backward()
            opt.step()
            avg_cost += cost / total_batch

            if i % 200 == 0:
                correct_prediction = (torch.max(preds.data, 1)[1] == Y.data)
                accuracy = correct_prediction.float().mean()
                print('Training Accuracy:', accuracy, cost)

        # 而 torch.cuda.empty_cache() 的作用就是释放缓存分配器当前持有的且未占用的缓存显存，以便这些显存可以被其他GPU应用程序中使用，并且通过 nvidia-smi命令可见。注意使用此命令不会释放tensors占用的显存。
        torch.cuda.empty_cache()
        end_time = time.time()
        epoch_mins, epoch_secs = data_utils.epoch_time(start_time, end_time)
        print("[Epoch: {:>4}] Time: {}m{}s| cost = {:>.9}".format(
            epoch + 1, epoch_mins, epoch_secs, avg_cost.data))

        Xq_val = Variable(strain_data.val_question_inps)
        Xa_val = Variable(strain_data.val_answer_inps)
        Y_val = Variable(strain_data.val_tgts)
        Qk_val = Variable(strain_data.val_Qk_inps)
        Ak_val = Variable(strain_data.val_Ak_inps)
        val_preds = encoder(Xq_val, Xa_val, Qk_val, Ak_val)
        val_cost = loss_function(preds, Y)
        # print(val_preds)
        correct_prediction_val = (
            torch.max(val_preds.data, 1)[1] == Y_val.data)
        accuracy_val = correct_prediction_val.float().mean()
        print('Validation Accuracy:', accuracy_val, val_cost)
    torch.save(encoder.state_dict(), 'ans_encoder.pt')
else:
    m_state = torch.load(model_path)
    encoder.load_state_dict(m_state)
print('Learning Finished!')

with torch.no_grad():
    encoder.eval()
    # print(self.encoder)

    Xq_eval = Variable(strain_data.eval_question_inps)
    Xa_eval = Variable(strain_data.eval_answer_inps)
    Y_eval = Variable(strain_data.eval_tgts)
    Qk_eval = Variable(strain_data.eval_Qk_inps)
    Ak_eval = Variable(strain_data.eval_Ak_inps)

    eval_preds = encoder(Xq_eval, Xa_eval, Qk_eval, Ak_eval)
    cost = loss_function(eval_preds, Y_eval)
    # print(F.softmax(eval_preds), eval_preds.size())
    correct_prediction = (torch.max(eval_preds.data, 1)[1] == Y_eval.data)
    accuracy = correct_prediction.float().mean()
    print('-----Evaluate Accuracy:', accuracy, cost)
    if os.path.exists(rank_path):
        model_ranking = []  # 排名
        model_scores = []  # 得分
        # zip() 函数⽤于将可迭代的对象作为参数，将对象中对应的元素打包成⼀个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
        for i, data_eval in zip(tqdm(range(len(evaluation_qa_pairs))), evaluation_qa_pairs):

            question = data_eval[0]  # 问题
            candidate_answers = data_eval[1]  # 候选答案
            Qk = data_eval[2]  # qk
            Ak = data_eval[3]  # ak
            ranks, scores = data_utils.rank_candidates(encoder, question, candidate_answers, Qk, Ak,
                                                       strain_data.word2index)  # 得到排名和得分
            # print('ranks:', ranks)
            # model_ranking.append([r[0] for r in ranks].index(0) + 1)
            # print(model_ranking)
            model_scores.append(scores)
            model_ranking.append([r[0] + 1 for r in ranks])  # 添加排名
        with open(rank_path, 'wb') as rank_file:
            pickle.dump([model_ranking, model_scores], rank_file)
    else:
        with open(rank_path, 'rb') as rank_file:
            model_ranking, model_scores = pickle.load(rank_file)
    #data_utils.dcg_score(model_ranking, k)
    for k in [1, 2, 3, 4, 5]:
        print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, data_utils.dcg_score(model_ranking, k),
                                                  k, data_utils.hits_count(model_ranking, k)))
