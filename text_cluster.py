#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:29:39 2018
提取招聘信息中的技能要求
以数据分析实习为例
文本聚类
@author: situ
"""

def loadDataset():
    '''导入文本数据集'''
    f = open('clean_text.txt','r',encoding="gbk")
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip())

    f.close()
    return dataset

text = loadDataset()

#关键词提取：用在GMM上
from jieba.analyse import extract_tags
extract_tags(" ".join(text), topK=15, withWeight=False, allowPOS=())


#词性标注
import jieba.posseg as pseg
words =pseg.cut(text[13])
for w in words:
    print(w.word,w.flag)


#NGram
from nltk import ngrams
from collections import Counter
import operator

def getNgram(text_i,n=2):
    analyzer2 = ngrams(text_i.split(),n)
    Ngram_dict = Counter(analyzer2)
    Ngram_dict
    ngram_list = []
    for k in Ngram_dict.keys():
        ngram_list.append("/".join(k))
    return ngram_list
    
text_2gram = list(map(getNgram,text))



with open("text_2gram.txt","w") as f2: #变成2-gram分词
    for sent in text_2gram:
        sent1 = " ".join(sent)+"\n"
        f2.write(sent1)


#1-2gram
from nltk import ngrams
from collections import Counter

def getNgram(text_i,n_range=[1,2]):
    analyzer1 = ngrams(text_i.split(),n_range[0])
    Ngram_dict1 = Counter(analyzer1)
    analyzer2 = ngrams(text_i.split(),n_range[1])
    Ngram_dict2 = Counter(analyzer2)
    Ngram_dict = Ngram_dict1+Ngram_dict2
    
    ngram_list = []
    for k in Ngram_dict.keys():
        if len(k)>1:
            ngram_list.append("/".join(k))
        else:
            ngram_list.append(k[0])
    return ngram_list
    
text_2gram = list(map(getNgram,text))



with open("text_2gram.txt","w") as f2:
    for sent in text_2gram:
        sent1 = " ".join(sent)+"\n"
        f2.write(sent1)





#余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2,use_idf=True)
X = vectorizer.fit_transform(dataset)
dist = 1 - cosine_similarity(X)

#kmeans tfidf 文本聚类---------------------------------------------------
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans,MiniBatchKMeans
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time



os.chdir("/Users/situ/Documents/EDA/final")
#os.chdir("E:/graduate/class/EDA/final")


def loadDataset(myfile):
    '''导入文本数据集'''
    f = open(myfile,'r',encoding = "utf-8")
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip())

    f.close()
    return dataset

def transform(dataset,n_features=1000):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2,use_idf=True)
    X = vectorizer.fit_transform(dataset)
    return X,vectorizer

def knn_train(X,vectorizer,true_k=10,minibatch = False,showLable = False):
    #使用采样数据还是原始数据训练k-means，    
    if minibatch:#数据多时用，如大于1万条样本
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=False)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                    verbose=False,random_state=1994)
    km.fit(X)    
    if showLable:
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print (vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :20]:
                print(' %s' % terms[ind], end='')
            print()
    result = list(km.predict(X))
    print ('Cluster distribution:')
    print (dict([(i, result.count(i)) for i in result]))
    return km
    
def test():
    '''测试选择最优参数'''
    dataset = loadDataset('clean_text.txt')    
    print("%d documents" % len(dataset))
    X,vectorizer = transform(dataset,n_features=500)
    true_ks = []
    scores = []
    for i in range(3,15,1):        
        score = -knn_train(X,vectorizer,true_k=i).score(X)/len(dataset)
        print (i,score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8,4))
    plt.plot(true_ks,scores,label="score",color="red",linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("score")
    plt.legend()
    plt.show()
    
def main():
    '''在最优参数下输出聚类结果'''
    dataset = loadDataset('clean_text.txt')
    X,vectorizer = transform(dataset,n_features=1000)
    km = knn_train(X,vectorizer,true_k=6,showLable=True)
    score = -km.score(X)/len(dataset)
    print (score)
    test()

if __name__ == '__main__':
    main()  

dataset = loadDataset('clean_text.txt')
X,vectorizer = transform(dataset,n_features=1000)    
start = time.time()
km = knn_train(X,vectorizer,true_k=6,showLable=True)  
end = time.time()
print("time cost:",end-start)  
km.labels_#第5类是技能要求

#关键词和权重
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
weights = np.sort(km.cluster_centers_)[:, ::-1]
#argsort()将每行数组的值从小到大排序后，并按照其相对应的索引值输出.
#argsort()[:, ::-1] 将每行数组的值从大到小排序后，并按照其相对应的索引值输出.
#np.sort(km.cluster_centers_)[:, ::-1] 把每行数组的值从大到小排序
terms = vectorizer.get_feature_names() #list

import csv
csvfile = open("term_weight_km.csv",'w',newline='',encoding='utf-8-sig') 
writer = csv.writer(csvfile)
for j in range(km.cluster_centers_.shape[0]):
    for i in range(500):
        writer.writerow([terms[order_centroids[j,i]],weights[j,i],j])
csvfile.close()

#GMM 文本聚类——————————————————————————————————————————————————————————————
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
import os
import numpy as np

#os.chdir("E:/graduate/class/EDA/final")
"""
tfidf 提取1000维特征时，GMM输出的概率不是0就是1
不知道是否降维后GMM可以输出概率？？？
"""
def loadDataset(myfile):
    '''导入文本数据集'''
    f = open(myfile,'r',encoding = "utf-8")
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip())

    f.close()
    return dataset

def transform(dataset,n_features=1000):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2,use_idf=True)
    X = vectorizer.fit_transform(dataset)
    return X,vectorizer

def gmm_train(X,vectorizer,n_components,showLable = False):
 
    gmmModel = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmmModel.fit(X.toarray())
    if showLable:
        print("Top terms per cluster:")
        order_centroids =gmmModel.means_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print (vectorizer.get_stop_words())
        for i in range(n_components):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :20]:
                print(' %s' % terms[ind], end='')
            print()
        result = list(gmmModel.predict(X.toarray())) #标签
        print ('Cluster distribution:')
        print (dict([(i, result.count(i)) for i in result]))
    return gmmModel
    
dataset = loadDataset('clean_text.txt')
X,vectorizer = transform(dataset,n_features=1000)   
start = time.time()
gmmModel = gmm_train(X,vectorizer,n_components=5,showLable=True)  #聚成7类以上是可以接受的结果 
end = time.time()
print("time cost:",end-start)  
gmm_labels = gmmModel.predict(X.toarray())#第3类是技能要求
#p = gmmModel.predict_proba(X.toarray())
#np.round(p,2)
#
#print(gmmModel.weights_)
#print(gmmModel.means_)
#print(gmmModel.covariances_)



#用jieba提取关键词
from jieba.analyse import extract_tags

def get_labels_kw(clean_text,labels,name, topK=15):
    def get_label_i(i,clean_text = clean_text,labels = labels): 
        """获取不同标签的文本"""
        j=0
        period_i=[]
        for j in list(range(len(clean_text))):
            if labels[j] == i:
                period_i.append(clean_text[j])
        print("标签%d有%d条文本"%(i,len(period_i)))
        return " ".join(period_i)

    def get_kw(text):
        return extract_tags(text, topK=topK, withWeight=False, allowPOS=())
    
    period_text = list(map(get_label_i,np.unique(labels)))
    news_kw = list(map(get_kw,period_text))

    for j in range(len(news_kw)):
        print("\n%s方法第%d个类别的关键词：\n"%(name,j+1))
        for i in range(len(news_kw[j])):
            print(news_kw[j][i])
 
get_labels_kw(dataset,gmm_labels,"gmm")    #第一类是技能要求

# NMF-based clustering-------------------------------------------------
#    ||X - UVT||
from sklearn.decomposition import NMF
from numpy.linalg import norm

def nmf_train(X,n_components):
    model = NMF(n_components=n_components, init='random', random_state=0)
    U = model.fit_transform(X.T)
    VT = model.components_    
    
    #1000个词，k = 10，文档数2772
    # 归一化
    V = VT.T
    
    nu,pu = U.shape
    nv,pv = V.shape
    
    for i in range(nv):
        for j in range(pv):
            V[i,j] = V[i,j]*norm(U[:,j])
    for j in range(pu):
        U[:,j] = U[:,j]/norm(U[:,j])
    #使用矩阵H来决定每个文档的归类。那个文档di的类标为m，当：m = argmaxj{vij}
    V.shape #(2773, 10)
    nmf_labels = list(map(np.argmax,V))
    return nmf_labels

from jieba.analyse import extract_tags

def get_labels_kw(clean_text,labels,name,topK=15):
    def get_label_i(i,clean_text = clean_text,labels = labels): 
        """获取不同标签的文本"""
        j=0
        period_i=[]
        for j in list(range(len(clean_text))):
            if labels[j] == i:
                period_i.append(clean_text[j])
        print("标签%d有%d条文本"%(i,len(period_i)))
        return " ".join(period_i)

    def get_kw(text):
        return extract_tags(text, topK=topK, withWeight=False, allowPOS=())
    
    period_text = list(map(get_label_i,np.unique(labels)))
    news_kw = list(map(get_kw,period_text))

    for j in range(len(news_kw)):
        print("\n%s方法第%d个类别的关键词：\n"%(name,j+1))
        for i in range(len(news_kw[j])):
            print(news_kw[j][i])
            
start = time.time()     
nmf_labels = nmf_train(X,n_components=5) #聚成5/6/7类的效果还行
get_labels_kw(dataset,nmf_labels,"nmf",20)   #第0类是技能要求  
end = time.time()
print("time cost:",end-start)  
#聚类效果比较——————————————————————————————————————————————————————————
# 兰德指数：比较聚类效果的相似性
from sklearn.metrics import adjusted_rand_score  
adjusted_rand_score(km.labels_,gmm_labels)    #相似度低。。
adjusted_rand_score(km.labels_,nmf_labels)     
adjusted_rand_score(gmm_labels,nmf_labels)       
# 轮廓系数
from sklearn.metrics import silhouette_score
silhouette_score(X,km.labels_, metric='euclidean')
silhouette_score(X,gmm_labels, metric='euclidean')
silhouette_score(X,nmf_labels, metric='euclidean')


# GAP 统计量        

#LDA-------------------------------------------------------------------
from gensim import corpora
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaModel


os.chdir("/Users/situ/Documents/EDA/final")


def loadDataset(myfile):
    '''导入文本数据集'''
    f = open(myfile,'r',encoding = "utf-8")
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip().split())

    f.close()
    return dataset


clean_text4 = loadDataset("clean_text.txt")
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
dictionary = corpora.Dictionary(clean_text4)
dictionary.save(os.path.join(TEMP_FOLDER, 'deerwester.dict'))  # store the dictionary, for future reference
print(dictionary)
print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in clean_text4]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  # store to disk, for later use
len(corpus)

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# Set training parameters.
num_topics = 5
#10 8
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)

top_topics = model.top_topics(corpus,5)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)


model.print_topic(1,30)
model.print_topic(3,30)

#判断一个训练集文档属于哪个主题
for index, score in sorted(model[corpus[0]], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, model.print_topic(index, 10)))
    
 
#给训练集输出其属于不同主题概率   
for index, score in sorted(model[corpus[0]], key=lambda tup: -1*tup[1]):
    print(index, score)
    
    
    
    
#判断一个测试集文档属于哪个主题
#unseen_document = [" ".join(text_i) for text_i in clean_text4[130]]
#unseen_document = " ".join(unseen_document)
    
unseen_document = text[130]
"""
还要对文档进行之前的文本预处理
"""


bow_vector = dictionary.doc2bow(unseen_document.split())
for index, score in sorted(model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, model.print_topic(index, 10)))

#给每个完整的分词后的文档生成不同主题的得分
import pandas as pd
import numpy as np
data = pd.read_csv("data_with_wordseg.csv",encoding = "gbk")
data.head()
lda_score = np.zeros((data.shape[0],num_topics))
for i in range(len(data["word_seg"])):
    line = data["word_seg"][i]
    bow_vector = dictionary.doc2bow(line.split())
    for index, score in sorted(model[bow_vector], key=lambda tup: -1*tup[1]):
        lda_score[i,index] = score
lda_score[:5,:]
lda_score_df = pd.DataFrame(lda_score,columns = ["lda"+str(i) for i in range(num_topics)])
lda_score_df.head()
data = pd.concat([data, lda_score_df], axis=1)
data.to_csv("data_with_lda_score.csv",index = False,encoding = "gbk")


"""    
探究不同主题得分与其他变量的关系    
"""
    
#LDA visualization---------------------------------------------------

import pyLDAvis
import pyLDAvis.gensim

vis_wrapper = pyLDAvis.gensim.prepare(model,corpus,dictionary)
pyLDAvis.display(vis_wrapper)
pyLDAvis.save_html(vis_wrapper,"lda%dtopics.html"%num_topics)


#pyLDAvis.enable_notebook()
#pyLDAvis.prepare(mds='tsne', **movies_model_data)






#新想法：
#把文本中的技能要求提取出来，然后再对技能类文本聚类，看看能不能聚出高端技能、低端技能
#再根据工资、地域、行业、技能进行聚类
lda_score = np.zeros((len(dataset),num_topics))
for i in range(len(dataset)):
    line = dataset[i]
    bow_vector = dictionary.doc2bow(line.split())
    for index, score in sorted(model[bow_vector], key=lambda tup: -1*tup[1]):
        lda_score[i,index] = score
lda_score[:5,:]
classify = pd.DataFrame(lda_score,columns = ["lda"+str(i) for i in range(num_topics)])
classify.head()




classify = pd.read_csv("clean_text_with_index.csv",encoding = "utf-8-sig")
classify.head()
#classify["text"] = dataset
classify["kmeans"] = km.labels_
classify["gmm"] = gmm_labels
classify["nmf"] = nmf_labels
#取并集选出技能主题

classify["skill"]=0
classify["skill"][classify["kmeans"]==5]=classify["skill"][classify["kmeans"]==5]+1
classify["skill"][classify["gmm"]==3]=classify["skill"][classify["gmm"]==3]+1
classify["skill"][classify["nmf"]==0]=classify["skill"][classify["nmf"]==0]+1
#classify["skill"][classify["lda8"]>0.5]=1


#classify.to_csv("text_labels.csv",index = False,encoding = "gbk")
# 手动把有关大数据、spark、hadoop、hive的样本选上


def skill_text_combine(df):
    """
    相同index的文本合并
    """
   
    index_list = df["index"].unique()
    skill_text = pd.DataFrame({"index":index_list,"skill_text":[""]*len(index_list)})
    for i in index_list:
        skill_text["skill_text"][skill_text["index"]==i] = " ".join(list(df["text"][df["index"]==i]))

    return skill_text

classify = pd.read_csv("text_labels.csv",encoding = "gbk")
skill_text = skill_text_combine(classify[classify["skill"]>0])
skill_text.to_csv("skill_text.csv",index = False,encoding = "gbk")

#一类只出现了msoffice
#一类还出现了sas spss python
#一类是大数据的软件相关
X,vectorizer = transform(skill_text["skill_text"],n_features=500)    
km = knn_train(X,vectorizer,true_k=3,showLable=True)    


gmmModel = gmm_train(X,vectorizer,n_components=3,showLable=True)   
gmm_labels = gmmModel.predict(X.toarray())
get_labels_kw(skill_text["skill_text"],gmm_labels,"gmm",20)

nmf_labels = nmf_train(X,n_components=3) 
get_labels_kw(skill_text["skill_text"],nmf_labels,"nmf",20)

skill_text["kmeans"] = km.labels_
skill_text["gmm"] = gmm_labels
skill_text["nmf"] = nmf_labels

skill_text["kmeans"] = skill_text["kmeans"].map({0:1,2:2,1:3})  
skill_text["gmm"] = skill_text["gmm"].map({1:1,0:2,2:3})  
skill_text["nmf"] = skill_text["nmf"].map({0:1,1:2,2:3})  
skill_text["score"] = np.mean(skill_text[["kmeans","gmm","nmf"]],axis=1)
skill_text.head()

#重新索引，与去重且重新索引后的data合并，看看能不能对应得上，如果有的职位描述没有专业技能句子，赋值0，有的话，根据三种聚类方式进行打分123取平均
data.tail()
#重新索引
data["index"]=list(range(len(data)))

data_with_skill = pd.merge(data,skill_text,how = "outer",on="index")
data_with_skill.head(10)
data_with_skill["score"] = data_with_skill["score"].fillna(0)
data_with_skill.to_csv("data_with_skill.csv",index = False,encoding = "gbk")
