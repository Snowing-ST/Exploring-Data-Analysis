# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:30:12 2018

@author: situ
"""

import numpy as np
import pandas as pd
import os
import re

#os.chdir("E:/graduate/class/EDA/final")
os.chdir("/Users/situ/Documents/EDA/final")
data = pd.read_csv("data_with_skill.csv",encoding = "gbk")
data.head()
data.info()

data.drop(["jobname","jobgood","url","city"],axis = 1,inplace = True)
#数值型数据处理----------------------
#每周工作天数
data.jobway.unique()
mapping = {}
for i in range(2,7):
    mapping[str(i) + '天／周'] = i
print(mapping)
data['day_per_week'] = data['jobway'].map(mapping)
data['day_per_week'].head()


#公司规模
data["size"].unique()
data["comp_size"] = ""
data["comp_size"][data['size'] == '少于15人'] = '小型企业'
data["comp_size"][data['size'] == '15-50人'] = '小型企业'
data["comp_size"][data['size'] == '50-150人'] = '中型企业'
data["comp_size"][data['size'] == '150-500人'] = '中型企业'
data["comp_size"][data['size'] == '500-2000人'] = '大型企业'
data["comp_size"][data['size'] == '2000人以上'] = '大型企业'

#实习月数
data.month.unique()
mapping = {}
for i in range(1,22):
    mapping["实习"+str(i) + '个月'] = i
print(mapping)
data['time_span'] = data['month'].map(mapping)
data['time_span'].apply(lambda f:int(f))

#每天工资
def get_mean_salary(s):
    return np.mean([int(i) for i in s[:(len(s)-2)].split("-")])
data['average_wage'] = data['salary'].apply(lambda s:get_mean_salary(s))
data['average_wage'].head()

data.drop(['jobway','size','month','salary'], axis = 1,inplace=True)

#字符型数据处理--------------------------------
#（城市）处理
#北京、上海、杭州、深圳、广州

def get_less_dummies(data,feature,useful_classes,prefix):
    useful_classes_prefix = [prefix+"_"+token for token in useful_classes]
    dum = pd.get_dummies(data[feature],prefix=prefix).ix[:,useful_classes_prefix]
    if sum(np.sum(dum.isnull()))>0:
        dum = dum.fillna(0)
    search_index = np.where(np.sum(dum,axis=1)==0)[0]
    for j in range(len(useful_classes)):
        token = useful_classes[j]
        for i in search_index:
            if len(re.findall(token,data.ix[i,feature]))>0:
                dum.ix[i,useful_classes_prefix[j]] = 1
#    print(dum.head())
    
    data = pd.concat([data,dum],axis = 1)
    return data

feature = "address"
useful_classes = ["北京","上海","杭州","深圳","广州","成都","武汉"]
data = get_less_dummies(data,feature,useful_classes,prefix="city")

#行业
#互联网，计算机，金融，电子商务和企业服务
 


feature = "industry"
useful_classes = ["互联网","计算机","金融","电子商务","企业服务","广告","文化传媒","电子","通信"]
data = get_less_dummies(data,feature,useful_classes,"industry")

data.head()


data.drop(['address','industry'], axis = 1,inplace=True)


#专业要求
def get_imp_info(data,feature,useful_classes,prefix):
    """直接从文本中提取"""
    useful_classes_prefix = [prefix+"_"+token for token in useful_classes]
    dum = pd.DataFrame(np.zeros((len(data),len(useful_classes))),columns = useful_classes_prefix)
    dum = dum.fillna(0)
    for j in range(len(useful_classes)):
        token = useful_classes[j]
#        print(token)
        for i in range(len(data)):
#            print(i)
            if len(re.findall(token,data.ix[i,feature].lower()))>0:
                dum.ix[i,useful_classes_prefix[j]] = 1
    print(dum.head())
    
#    data = pd.concat([data,dum],axis = 1)
    return dum


feature = "contents"
useful_classes = ["统计","计算机","数学"]
dum = get_imp_info(data,feature,useful_classes,"subject")
data = pd.concat([data,dum],axis = 1)
data.head()

#技能要求
def get_imp_info2(data,feature,useful_classes,prefix):
    """从分词中提取"""
    useful_classes_prefix = [prefix+"_"+token for token in useful_classes]
    dum = pd.DataFrame(np.zeros((len(data),len(useful_classes))),columns = useful_classes_prefix)
    dum = dum.fillna(0)
    for j in range(len(useful_classes)):
        token = useful_classes[j]
#        print(token)
        for i in range(len(data)):
            word_list = data.ix[i,feature].split()
            if token in word_list:
                print(data.ix[i,feature])
                dum.ix[i,useful_classes_prefix[j]] = 1
    print(dum.head())
    
#    data = pd.concat([data,dum],axis = 1)
    return dum


feature = "contents"
#useful_classes = ["python","r语言","spss","excel","ppt","word","sql","sas","vba","office","msoffice",
#                  "hadoop","spark","hive","scala","hbase","java","matlab","linux","shell","c#"]
#                  "机器学习","数据挖掘","数学建模","自然语言处理","自然语言","文本挖掘",
useful_classes = ['excel', 'sql', 'python', 'sas', 'spss','hadoop', 'spark', 'hive', 'shell', 'java']                  
dum = get_imp_info(data,feature,useful_classes,"skill")
np.sum(dum)
# 技能要求前10：excel sql python sas spss | hadoop spark hive shell java 
data = pd.concat([data,dum],axis = 1)
data.head()

#技能与平均薪资
def mean_salary(useful_classes,data,salary,prefix):
    feature_list = [prefix+"_"+skill for skill in useful_classes]
    p = len(feature_list)
    df = pd.DataFrame(np.zeros((p,3)),columns = ["skill","mean_salary","count"])
    df["skill"] = useful_classes
    for i in range(p):
        df["mean_salary"][df["skill"]==useful_classes[i]] = np.mean(data[salary][data[feature_list[i]]==1])
        df["count"][df["skill"]==useful_classes[i]] = len(data[salary][data[feature_list[i]]==1])
    return df

useful_classes = ['excel', 'sql', 'python', 'sas', 'spss','hadoop', 'spark', 'hive', 'shell', 'java']                  
salary = "average_wage"
prefix = "skill"
df = mean_salary(useful_classes,data,salary,prefix)

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
plt.figure(figsize=(8,5)) 
sns.stripplot(x = "skill",y="mean_salary",data=df,size = 10)
plt.xlabel("skill_software")
plt.ylabel("mean_salary")
plt.savefig("skill_salary.jpg")

# 公司
data["compname"].value_counts()


data.drop(['compname'], axis = 1,inplace=True)
#data = pd.get_dummies(data)

#data.to_csv("data_analysis.csv",index = False,encoding = "gbk")


from sklearn.linear_model import LinearRegression
X = data.drop(["average_wage",'contents','kmeans','gmm','nmf',"skill_text","index","compname"],axis = 1);Y = data["average_wage"]
X = pd.get_dummies(X)
regr = LinearRegression().fit(X,Y)
#输出R的平方
print(regr.score(X,Y))
regr.coef_




#职位诱惑可以做词云图