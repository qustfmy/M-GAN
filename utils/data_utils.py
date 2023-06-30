import re
import os
import pickle
import numpy as np
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import dcg_score
from sentence_transformers import SentenceTransformer, util

"""
文本准备
"""


def text_prepare(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\\[]@,;]')
    GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = text.lower()  # 转换字符串中所有大写字符为小写。
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = GOOD_SYMBOLS_RE.sub('', text)
    # 将序列中的元素以指定的字符连接生成一个新的字符串。
    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS])
    return text.strip()


"""
查询向量
"""


def query_to_vec(query, embeddings, dim=300):
    vec = np.zeros((dim,), dtype=np.float32)
    count = 0
    for w in query.split():
        if w in embeddings:
            count += 1
            vec += embeddings[w]
    if count == 0:
        return vec
    return vec / count


"""
从语料中去前k个最相似
"""


def semblance(q_data, query, K):
    dictionary = corpora.Dictionary(q_data)  # 是文档集的表现形式。corpora就是一个二维矩阵
    doc_vectors = [dictionary.doc2bow(text) for text in q_data]  # 建立语料库
    doc_text_vec = dictionary.doc2bow(query)
    tfidf = models.TfidfModel(doc_vectors)
    index = similarities.SparseMatrixSimilarity(
        tfidf[doc_vectors], num_features=len(dictionary.keys()))  # 文本相似度计算
    sim = index[tfidf[doc_text_vec]]
    sim_sorted = sorted(enumerate(sim, 1), key=lambda x: x[1], reverse=True)

    return sim_sorted[:K]


"""
:param 文件路径
"""


def read_data(data_file):
    X = []  # qa对
    Y = []  # 标签
    Z = []  # qk 和ak
    K = 10
    qa_best = []
    q_best = []

    with open(data_file, 'r', encoding='utf-8') as data_file:
        for line in data_file:

            pid, question, answer, label = line.split('\t')  # 切分出标号，问题，答案，标签
            question = text_prepare(question)
            answer = text_prepare(answer)
            # prediction = text_prepare(prediction)

            # question = question + " " + prediction
            qa_pair = (question, answer)  # 形成字典
            label = int(label)

            if label == 1:  # 如果是正样本
                qa_best.append(qa_pair)
                q_best.append(question.split())  # 问题答案加入
            X.append(qa_pair)
            Y.append(label + 2)
        print('build dict')
        dictionary = corpora.Dictionary(q_best)  # 建立向量
        doc_vectors = [dictionary.doc2bow(text) for text in q_best]  # 建立语料库
        tfidf = models.TfidfModel(doc_vectors)
        index = similarities.SparseMatrixSimilarity(
            tfidf[doc_vectors], num_features=len(dictionary.keys()))  # 文本相似度计算

        print('----------------------')
        print('start find sim_question')
        for i in tqdm(range(len(X))):  # 进度条
            QA = []
            q = X[i][0].split()
            doc_text_vec = dictionary.doc2bow(q)
            sim = index[tfidf[doc_text_vec]]
            sim_sorted = sorted(enumerate(sim, 1),
                                key=lambda x: x[1], reverse=True)
            sim_index = sim_sorted[:K]
            for j in range(len(sim_index)):
                QA.append(qa_best[sim_index[j][0] - 1])
            Z.append(QA)
        print('finish find sim_question')
    X, Y, Z = shuffle(X, Y, Z, random_state=0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    # 返回X：QA对 Y：是标签 Z是qk和ak
    return X, Y, Z, qa_best, q_best


def read_data2(data_file):
    X = []  # qa对
    Y = []  # 标签
    Z = []  # qk 和ak
    K = 10
    qa_best = []
    q_best = []
    q_text_list = []
    a_text_list = []
    q_best_text_list = []
    dict = {}
    with open(data_file, 'r', encoding='utf-8') as data_file:
        for line in data_file:

            pid, question, answer, label = line.split('\t')  # 切分出标号，问题，答案，标签
            question = text_prepare(question)
            answer = text_prepare(answer)
            q_text_list.append(question)
            a_text_list.append(answer)
            qa_pair = (question, answer)  # 形成字典
            label = int(label)

            if label == 1:  # 如果是正样本
                q_best_text_list.append(question)
                qa_best.append(qa_pair)
                q_best.append(question.split())  # 问题答案加入
            X.append(qa_pair)
            Y.append(label + 2)
        print('build dict')
        for i in range(0, len(q_text_list)):
            dict.update({q_text_list[i]: a_text_list[i]})
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        corpus_embedding = model.encode(
            q_best_text_list, device='cuda', show_progress_bar=True)
        query_embedding = model.encode(
            q_text_list, device='cuda', show_progress_bar=True)
        cos_scores = util.semantic_search(
            query_embedding, corpus_embedding, top_k=K+1)
        for index, quest in tqdm(enumerate(q_text_list)):
            # 将1到10的的corpus_id取到
            # 选取q_text_list的问答对
            id_list = []
            cand_list = []
            for i in range(1, 11):
                id_list.append(cos_scores[index][i]['corpus_id'])
            for index1, i in enumerate(id_list):
                pair = (q_text_list[i], dict[q_text_list[i]])
                cand_list.append(pair)
            Z.append(cand_list)
        print('finish find sim_question')
    X, Y, Z = shuffle(X, Y, Z, random_state=0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    return X, Y, Z, qa_best, q_best


def read_file(data_file):
    with open(data_file, 'r', encoding='utf-8') as data:
        X = []
        Y = []
        Z = []
        for line in data:
            pid, qcq, answer, QAK, label = line.split('\t')
            X.append((qcq, answer))
            Y.append(int(label))
            Z.append(QAK)
    X, Y, Z = shuffle(X, Y, Z, random_state=0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    # X:问题和答案对
    # Y:标签
    # Z:是qk和ak
    return X, Y, Z


"""
分割数据
X:param 训练数据
"""


def split_data(X, Z):
    Xq = X[:, 0]
    Xa = X[:, 1]
    Qk = Z[:, :, 0]
    Ak = Z[:, :, 1]
    # Xq：问题
    return Xq, Xa, Qk, Ak


"""
建立词表
self.Xqtrain, self.Xatrain, self.Xqval, self.Xaval
"""


def build_vocab(Xqtrain, Xatrain, Xqval, Xaval):
    vocab = []

    for e in Xqtrain:
        for word in e.split():
            vocab.append(word)

    for e in Xatrain:
        for word in e.split():
            vocab.append(word)

    for e in Xqval:
        for word in e.split():
            vocab.append(word)

    for e in Xaval:
        for word in e.split():
            vocab.append(word)

    vocab = set(vocab)  # 变为集合
    return vocab

# Y_train 标签 ，vocab


def build_dict(Y_train, vocab):
    word2index = {'pad': 0, 'unk': 1}
    # hello
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)  # {'pad': 0, 'unk': 1,'hello':2,}
    index2word = {v: k for k, v in word2index.items()}  # 做颠倒

    target2index = {}

    for cl in set(Y_train):
        if target2index.get(cl) is None:
            target2index[cl] = len(target2index)  # {'1': 0, '2': 1,}

    index2target = {v: k for k, v in target2index.items()}
    vocab_size = len(word2index)

    return word2index, index2word, index2target, vocab_size


"""
xq, word2index
准备句子
"""


def prepare_sequence(seq, word2index):
    idxs = []
    for w in seq.split():
        if w in word2index:
            idxs.append(word2index[w])
        else:
            idxs.append(word2index["unk"])
    # return Variable(LongTensor(idxs))
    return idxs


"""
xq, word2index
准备句子
"""


def pad_data(Xq_data, Xa_data, Qk_data, Ak_data, word2index):
    print('pad Xqa data')
    Xq_p, Xa_p = [], []
    for i, xq, xa in zip(tqdm(range(len(Xq_data))), Xq_data, Xa_data):
        Xq_p.append(prepare_sequence(xq, word2index))
        Xa_p.append(prepare_sequence(xa, word2index))
    Xq = pad_sequences(Xq_p, maxlen=32)
    Xa = pad_sequences(Xa_p, maxlen=128)

    print('pad QAK data')
    first = True
    for i, qk, ak in zip(tqdm(range(len(Qk_data))), Qk_data, Ak_data):
        Q = []
        A = []
        for q_t, a_t in zip(qk, ak):
            Q.append(prepare_sequence(q_t, word2index))
            A.append(prepare_sequence(a_t, word2index))
        Qk_p = pad_sequences(Q, maxlen=32)
        Qk_p = Qk_p[np.newaxis, :]
        Ak_p = pad_sequences(A, maxlen=128)
        Ak_p = Ak_p[np.newaxis, :]
        if first:
            Qk = Qk_p
            Ak = Ak_p
            first = False
        else:
            Qk = np.vstack([Qk, Qk_p])
            Ak = np.vstack([Ak, Ak_p])
    return Xq, Xa, Qk, Ak

# glove模型 单词转词向量


def star_embdding(file):
    starspace_embeddings = {}
    for line in open(file, 'r', encoding='ISO-8859-1'):
        try:
            word, *vec = line.strip().split()
            vf = []
            for v in vec:
                vf.append(float(v))
            starspace_embeddings[word] = np.array(vf)
        except:
            continue
    return starspace_embeddings

# 推荐的评价指标


def hits_count(candidate_ranks, k):
    count = 0
    for rank in candidate_ranks:
        for r in rank[:k]:
            if r == 1:
                count += 1
    print(count)
    return count / (len(candidate_ranks) + 1e-8)

# 推荐的评价指标


def dcg_score(candidate_ranks, k):
    score = 0
    for rank in candidate_ranks: #为1是
        for i in range(k):  # i出错
            if rank[i] == 1:
                score += 1 / np.log2(2 + i)
    # print(score)
    return score / (len(candidate_ranks) + 1e-8)

# def new_dcg(candidate_ranks, model_scores, k):
#     for k in candidate_ranks:
#         score+=dcg_score(candidate_ranks[],model_scores)
#     print(score)
#     return score / (len(candidate_ranks) + 1e-8)

#MRR评价指标


def top_k(candidate_ranks, k):
    count = 0
    values, indices = torch.topk(
        candidate_ranks, k, dim=1, largest=True, sorted=True)
    # print(indices)
    for indice in indices:
        if 0 in indice:
            count += 1
    # print('count:', count)
    return count / (len(candidate_ranks) + 1e-8)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 位置编码 没用到


def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * \
                (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

# 验证 train——data——4cl  va——data——4cl e不带
# Variable是你第二步
# eval


def eval_Data(qa_best, q_best, evaluate_data_path):
    # 正样本答案对 语料
    if not os.path.exists(evaluate_data_path):
        evaluation_qa_pairs = []
        K = 10
        dictionary = corpora.Dictionary(q_best)
        doc_vectors = [dictionary.doc2bow(text) for text in q_best]
        tfidf = models.TfidfModel(doc_vectors)
        index = similarities.SparseMatrixSimilarity(
            tfidf[doc_vectors], num_features=len(dictionary.keys()))
        # ./notebook/datastap/data/test_data.csv
        # ./subdata/Q2CQ/evaluation_data
        with open('./subdata/Q2CQ/evaluation_data', 'r') as evaluation_data:
            for line in evaluation_data:
                Qk = []
                Ak = []
                qa = line.strip().split('\t')  # 得到yihang
                question = text_prepare(qa[0])
                # print(question)
                # prediction = qa[1]
                # question = question + ' '.join(set(text_prepare(prediction).split()))

                doc_text_vec = dictionary.doc2bow(question.split())
                sim = index[tfidf[doc_text_vec]]
                sim_sorted = sorted(enumerate(sim, 1),
                                    key=lambda x: x[1], reverse=True)
                sim_index = sim_sorted[:K]
                for j in range(len(sim_index)):
                    sim_qa = qa_best[sim_index[j][0] - 1]
                    Qk.append(sim_qa[0])
                    Ak.append(sim_qa[1])
                # candidate_answers = qa[2:7]
                candidate_answers = [text_prepare(a) for a in qa[1:]]
                evaluation_qa_pairs.append(
                    (question, candidate_answers, Qk, Ak))
            evaluation_qa_pairs = shuffle(evaluation_qa_pairs, random_state=0)
        with open(evaluate_data_path, 'wb') as e_data:
            pickle.dump(evaluation_qa_pairs, e_data)
    else:
        with open(evaluate_data_path, 'rb') as e_data:
            evaluation_qa_pairs = pickle.load(e_data)
    return evaluation_qa_pairs

# 排名候选


def eval_Data2(qa_best, q_best, evaluate_data_path):
    # 正样本答案对 语料
    if os.path.exists(evaluate_data_path):
        evaluation_qa_pairs = []
        K = 10
        dictionary = corpora.Dictionary(q_best)
        doc_vectors = [dictionary.doc2bow(text) for text in q_best]
        tfidf = models.TfidfModel(doc_vectors)
        index = similarities.SparseMatrixSimilarity(
            tfidf[doc_vectors], num_features=len(dictionary.keys()))

        with open('./notebook/secord/data/test_data.csv', 'r') as evaluation_data:
            for i, line in tqdm(enumerate(evaluation_data)):
                Qk = []
                Ak = []
                qa = line.strip().split('\t')  # 得到yihang
                question = text_prepare(qa[1])
                doc_text_vec = dictionary.doc2bow(question.split())
                sim = index[tfidf[doc_text_vec]]
                sim_sorted = sorted(enumerate(sim, 1),
                                    key=lambda x: x[1], reverse=True)
                sim_index = sim_sorted[:K]
                for j in range(len(sim_index)):
                    sim_qa = qa_best[sim_index[j][0] - 1]
                    Qk.append(sim_qa[0])
                    Ak.append(sim_qa[1])
                # candidate_answers = qa[2:7]
                candidate_answers = [text_prepare(a) for a in qa[2:]]
                evaluation_qa_pairs.append(
                    (question, candidate_answers, Qk, Ak))
            evaluation_qa_pairs = shuffle(evaluation_qa_pairs, random_state=0)
        with open(evaluate_data_path, 'wb') as e_data:
            pickle.dump(evaluation_qa_pairs, e_data)
    else:
        with open(evaluate_data_path, 'rb') as e_data:
            evaluation_qa_pairs = pickle.load(e_data)
    return evaluation_qa_pairs


def rank_candidates(encoder, question, candidate_answers, Qk, Ak, word2index):
    candidate_scores = []
    for answer in candidate_answers:

        Xq_p, Xa_p = [], []
        Xq_p.append(prepare_sequence(question, word2index))  # 得到句子数字表示
        Xa_p.append(prepare_sequence(answer, word2index))

        Xq_val = torch.LongTensor(pad_sequences(
            Xq_p, maxlen=32)).cuda()  # 转成向量格式
        Xa_val = torch.LongTensor(pad_sequences(Xa_p, maxlen=128)).cuda()

        Q = []
        A = []
        for q_t, a_t in zip(Qk, Ak):
            Q.append(prepare_sequence(q_t, word2index))  # 候选答案Q
            A.append(prepare_sequence(a_t, word2index))
        Qk_p = pad_sequences(Q, maxlen=32)  # 将序列转化为经过填充以后的一个长度相同的新序列新序列
        Qk_val = Qk_p[np.newaxis, :]  # 在np.newaxis这里增加1维
        Ak_p = pad_sequences(A, maxlen=128)
        Ak_val = Ak_p[np.newaxis, :]
        Qk_val = torch.LongTensor(Qk_val).cuda()
        Ak_val = torch.LongTensor(Ak_val).cuda()
        Xq_val = Variable(Xq_val)
        Xa_val = Variable(Xa_val)
        Qk_val = Variable(Qk_val)
        Ak_val = Variable(Ak_val)
        preds = encoder(Xq_val, Xa_val, Qk_val, Ak_val)
        # print('preds:', preds)
        # 将模拟生成的答案特征向量作为判别候选答案的依据，将两者拼接后经过全连接层，将结果维度转换为对应于数据4种标签的4维向量，最后使用softmax计算其分布在4种标签上的概率。的计算公式如下：
        preds = F.softmax(preds, dim=1)
        #print('preds:', preds)
        np_prediction = preds.detach().cpu().numpy()  # 得到numpy数组表示
        # 计算最终生成答案得分
        score = (-2. * np_prediction[0][0]) + (-1. * np_prediction[0][1]) + (1. * np_prediction[0][2]) + (
            2. * np_prediction[0][3])
        # score = np_prediction[0][3]
        candidate_scores.append(score)

    tl = [(i, candidate_answers[i], candidate_scores[i])
          for i in range(len(candidate_answers))]  # 列表[]

    # key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。true降序
    stl = sorted(tl, key=lambda x: x[2], reverse=True)
    result = [(t[0], t[1]) for t in stl]
    # result是一个列表，列表中的元素是元组，元组中的第一个元素是答案在候选答案中的索引，第二个元素是答案
    return result, candidate_scores
