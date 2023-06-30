import pymongo, joblib, re
from tqdm import tqdm
import numpy as np
import pandas as pd

from tqdm import tqdm

qa_set = pd.read_csv("./data/qa_set_ori.csv")
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
new_qa.to_csv('./data/qa_set_u0.csv')