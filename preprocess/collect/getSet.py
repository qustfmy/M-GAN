import pymongo, joblib, re
from tqdm import tqdm
import numpy as np
import pandas as pd

# 用户Ux<tab>问题Qy<tab>回答Axy<tab>得分Rxy的文件
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["qa_2017"]
mycol = mydb["qa_2017"]
# 对一串文本进行粗处理：
# 去掉换行 小写替换大写 去掉html标签 连续的空格替换为一个空格
# 返回这个文本
def textPreProcess(content):
    content = str(content).replace("\n", " ").lower()
    content = "".join(re.sub(u'\<.*?\>', '', content))
    content = ' '.join(content.split())
    return content

data_all = mycol.find({},
               {'q_question_id': 1, 'q_title': 1, 'q_body': 1, 'answers': 1, 'q_creation_date': 1, 'q_is_answered': 1,
                'q_accepted_answer_id': 1})
qa_set = pd.DataFrame()
for curr_question in tqdm(data_all):
    curr_question = dict(curr_question)
    q_id = str(curr_question['q_question_id'])  # 当前问题的id
    q_time = curr_question['q_creation_date']   # 问题创建时间
    answers = curr_question['answers']          # 回答的答案
    q_is_answered = curr_question['q_is_answered']  # 问题是否被回答
    q_accepted_answer_id = curr_question['q_accepted_answer_id']    # 问题的被接受的回答id
    q_content = str(curr_question['q_title'] + curr_question['q_body'])  # 当前问题的内容
    q_content = textPreProcess(q_content)
    if q_accepted_answer_id != None:
        if answers==None:
            continue
        if len(answers) > 0:
            answer_count = len(answers)
            for answer in answers:
                if answer['owner']['user_type'] != 'registered':
                    continue
                u_id = answer['owner']['user_id']
                a_time = answer['creation_date']
                a_id = answer['answer_id']
                a_content = str(answer['body'])  # 当前回答的内容
                a_content = textPreProcess(a_content)  # 去掉换行 去掉<>标签 转换成小写
                score = answer['score']  # 当前回答的得分情况
                if a_id == q_accepted_answer_id:
                    r = [1.0, 0.0, 0.0, 0.0]
                else:
                    r = [0.0, 1.0, 0.0, 0.0]
                item = pd.DataFrame()
                item['q_id'] = [q_id]
                item['q_time'] = [q_time]
                item['q_text'] = [q_content]
                item['u_id'] = [u_id]
                item['a_id'] = [a_id]
                item['a_text'] = [a_content]
                item['a_time'] = [a_time]
                item['r'] = [r]
                item['score'] = [score]
                # print(item)
                qa_set = pd.concat([qa_set, item])
qa_set = qa_set.dropna(axis=0, how='any')
qa_set.to_csv("qa_set_ori.csv")     # 存在回答数量较少的用户，后面根据用户回答数量进行筛选选出4000左右的用户






from tqdm import tqdm

qa_set = pd.read_csv("./GetDataSet3/qa_set_ori.csv")
u_id_group = qa_set.groupby("u_id")
print("原始问答数据读取完成...")
user_count = 0
qa_count = 0
new_qa = pd.DataFrame()
for user in tqdm(u_id_group):
    if len(user[1]) > 0:  # 替换为0-30的数字做统计
        user_count += 1
        qa_count += len(user[1])
        new_qa = pd.concat([new_qa, user[1]])
print(" >0 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
new_qa.to_csv('qa_set_u0.csv')