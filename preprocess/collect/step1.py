import pymongo, joblib, re
from tqdm import tqdm
import numpy as np
import pandas as pd

# 用户Ux<tab>问题Qy<tab>回答Axy<tab>得分Rxy的文件
client = pymongo.MongoClient("localhost",27017)

#选择数据库 ↓
#获取集合 ↓
mydb = client["qa_2017"]
mycol = mydb["qa_2017"]
#获取collection1集合内全部数据 ↓
# x = collection1.find()


def textPreProcess(content):
    content = str(content).replace("\n", " ").lower()
    content = "".join(re.sub(u'\<.*?\>', '', content))
    content = ' '.join(content.split())
    return content
def textCodeProcess(startStr,endStr,string):
    result = re.findall(startStr + "(.*?)" + endStr, string)
    # result  是筛选出来的中间字符列表格式
    # 中间字符转换成字符串
    middleStr = "".join(result)
    # 中间字符拼接成sub函数可以批量截取的格式
    middleStr = "[" + middleStr + "]"
    # print(re.sub(middleStr,'',string))
    middleStr = re.sub(middleStr, '', string)
    middleStr = re.sub(startStr + endStr, '', middleStr)
    return middleStr
def clearContentWithSpecialCharacter(content):
    # 先将<!--替换成，普通字符l
    content = content.replace("<code>", "l")
    # 再将-->替换成，普通字符l
    content = content.replace("</code>", "l")
    # 分组标定，替换，
    pattern = re.compile(r'(l)(.*)(l)')
    # 如果想包括两个1，则用pattern.sub(r\1''\3,content)
    return pattern.sub(r'', content)
def deleteByStartAndEnd(s, start, end,count):
    # 找出两个字符串在原始字符串中的位置，开始位置是：开始始字符串的最左边第一个位置，结束位置是：结束字符串的最右边的第一个位置
    if count ==0:
        return s
    else:
        x1 = s.find(start)
        x2 = s.find(end) + len(end)  # s.index()函数算出来的是字符串的最左边的第一个位置
        # 找出两个字符串之间的内容
        x3 = s[x1:x2]
        # 将内容替换为控制符串
        result = s.replace(x3, "")
        return result
data_all = mycol.find({},
               {'q_question_id': 1, 'q_title': 1, 'q_body': 1, 'answers': 1, 'q_creation_date': 1, 'q_is_answered': 1,
                'q_accepted_answer_id': 1})
qa_set = pd.DataFrame()
# count=0;
for curr_question in tqdm(data_all):
    # count+=1
    # if count==10:
    #     break;
    curr_question = dict(curr_question)
    q_id = str(curr_question['q_question_id'])  # 当前问题的id
    q_time = curr_question['q_creation_date']   # 问题创建时间
    answers = curr_question['answers']          # 回答的答案
    q_is_answered = curr_question['q_is_answered']  # 问题是否被回答
    q_accepted_answer_id = curr_question['q_accepted_answer_id']    # 问题的被接受的回答id
    q_content = str(curr_question['q_title'] + curr_question['q_body'])  # 当前问题的内容
    q_count=q_content.count('<code>') #得到问题中code的次数
    for i in range(0,q_count):
        q_content= deleteByStartAndEnd(q_content, "<code>", "</code>",q_count)
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
                a_count = a_content.count('<code>')  # 得到问题中code的次数
                for i in range(0, q_count):
                    a_content = deleteByStartAndEnd(a_content, "<code>", "</code>",a_count)
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
qa_set = qa_set.dropna(axis=0, how='any') #去除含有NaN的行
qa_set.to_csv("./data/qa_set_ori.csv")

client.close()
