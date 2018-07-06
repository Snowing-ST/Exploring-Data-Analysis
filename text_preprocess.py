# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:55:25 2018
提取招聘信息中的技能要求
以数据分析实习为例

@author: situ
"""

import pandas as pd
import numpy as np
import os
import re
import jieba
from collections import Counter,defaultdict
import operator
from nltk import ngrams
import csv
import matplotlib.pyplot as plt

"""
自定义词典选取词的方式：
1. 查看分词结果
2. 查看2-gram词频统计
"""
#os.chdir("E:/graduate/class/EDA/final")
os.chdir("/Users/situ/Documents/EDA/final")
jieba.load_userdict("dict.txt")



#手动删除英文的招聘信息
data = pd.read_csv("数据分析_共47页.csv",encoding = "gbk")
data.head()
text = data["contents"]

#删除重复内容
sum(data["contents"].duplicated()) 
data[data["contents"].duplicated()]

data = data.drop_duplicates(["contents"])
#检查是否有空值
sum(data["contents"].isnull())


#文本预处理————————————————————————————————————————————————————————
def get_text(data):
    text=data["contents"]
    text = text.dropna() 
    len(text)
    text=[t.encode('utf-8').decode("utf-8") for t in text] 
    return text
def get_stop_words(file='stopWord.txt'):
    file = open(file, 'rb').read().decode('utf8').split(',')
    file = [line.strip() for line in file]
    return set(file)                                         #查分停用词函数


def rm_tokens(words):                                        # 去掉一些停用词和完全包含数字的字符串
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:                      # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list



def rm_char(text):

    text = re.sub('\x01', '', text)                        #全角的空白符
    text = re.sub('\u3000', '', text) 
    text = re.sub(r"[\)(↓%·▲ \s+】&【]","", text) 
    text = re.sub(r"[\d（）《》><‘’“”"".,-]"," ",text,flags=re.I)
    text = re.sub('\n+', " ", text)
    text = re.sub('[，、：。！？?；——]', " ", text)
    text = re.sub(' +', " ", text)
    return text

def convert_doc_to_wordlist(paragraph, cut_all=False):
    sent_list = [sent for sent in re.split(r"[。！;:\n.；：?]",paragraph)]
    sent_list = map(rm_char, sent_list)                       # 去掉一些字符，例如\u3000
    word_2dlist = [rm_tokens(jieba.cut(part, cut_all=cut_all))
                   for part in sent_list]                     # 分词
#    word_list = sum(word_2dlist, [])
    def rm_space_null(alist):
        alist = [s for s in alist if s not in [""," "]]
        return alist
    rm_space = [rm_space_null(ws) for ws in word_2dlist if len(ws)>0]
    return rm_space


def rm_1ow_freq_word(texts,low_freq=1):
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [[token for token in text if frequency[token] > low_freq] for text in texts]
    return texts

def rm_short_len_word(texts,short_len=0):
    texts = [[token for token in text if len(token)>short_len and len(token)<15] for text in texts]
    return texts



def rm_high_freq_word(texts,num=10,other_dele_file="delete_words.txt"):
    whole_text = []
    for doc in texts:
        whole_text.extend(doc)
    word_count = np.array(Counter(whole_text).most_common())
    high_freq = []
    for i in range(num):
        high_freq.append(word_count[i][0])
    if other_dele_file!=None:
        other_dele_list = open(other_dele_file, 'rb').read().decode('gbk').split('\n')
        high_freq.extend(other_dele_list)
        dele_list = np.unique(high_freq)
    else:
        dele_list = high_freq
#    print(dele_list)
    texts = [[token.lower() for token in text if token not in dele_list] for text in texts]
    return texts



"""
可尝试词性标注，把动词去掉，没试，感觉不好
尝试2-Gram模型，已尝试，效果不好
"""
    
def word_seg():
    """
    对职位描述进行分词，保存在原来的文件里
    """
    clean_text=[convert_doc_to_wordlist(line) for line in get_text(data)]
    clean_text[0]
    clean_text_for_wordseg =[ " ".join([" ".join(sentlist)  for sentlist in line]) for line in clean_text]
    clean_text_for_wordseg = [line.split() for line in clean_text_for_wordseg]
    
    clean_text_for_wordseg = rm_high_freq_word(rm_short_len_word(clean_text_for_wordseg))
    data["word_seg"] = [" ".join(line) for line in clean_text_for_wordseg]
    data.to_csv("data_with_wordseg.csv",index = False,encoding = "gbk")
    
    
def main():
    clean_text=[convert_doc_to_wordlist(line) for line in get_text(data)]     
    clean_text = [sent for para in clean_text for sent in para] #拆成一个个句子
    
#    length = np.array([len(sent) for sent in clean_text]) #检查太长的文本
#    plt.hist(length)
#    np.array(clean_text)[length>50]
    
    
    clean_text2 = rm_1ow_freq_word(clean_text)
    clean_text3 = rm_short_len_word(clean_text2)
    clean_text4 = rm_high_freq_word(clean_text3)
    clean_text5 = [sent for sent in clean_text4 if len(sent)>2]
#    len(clean_text4)
    
    with open("clean_text.txt","w") as f2:
        for sent in clean_text5:
            sent1 = " ".join(sent)+"\n"
            f2.write(sent1)


if __name__ == '__main__':
    main()  


#保留index的clean_text
clean_text=[convert_doc_to_wordlist(line) for line in get_text(data)] 

sent_index = [[i]*len(clean_text[i]) for i in range(len(clean_text))]
sent_index = [index for index_set in sent_index for index in index_set]    
len(sent_index)
clean_text = [sent for para in clean_text for sent in para] #拆成一个个句子
clean_text2 = rm_1ow_freq_word(clean_text)
clean_text3 = rm_short_len_word(clean_text2)
clean_text4 = rm_high_freq_word(clean_text3)
clean_text5 = [sent for sent in clean_text4 if len(sent)>2]

sent_index5 = [sent_index[i] for i in range(len(clean_text4)) if len(clean_text4[i])>2]        

clean_text5 = [" ".join(sent_list) for sent_list in clean_text5]
clean_text_with_index = pd.DataFrame({"index":sent_index5,"text":clean_text5})
clean_text_with_index.to_csv("clean_text_with_index.csv",index = False,encoding = "utf-8-sig")


csvfile = open("clean_text_with_index.csv",'w',newline='',encoding='utf-8-sig') 
writer = csv.writer(csvfile)
for i in range(len(clean_text5)):
    writer.writerow([sent_index5[i]+2," ".join(clean_text5[i])])
csvfile.close()





#把分词前后对比保存进csv
sent_list = [sent for paragraph in get_text(data) for sent in re.split(r"[。！;:\n.；：]",paragraph)]
sent_list_2 = [sent_i for sent_i in sent_list if len(sent_i)>5]

sent_list_2 = list(map(rm_char, sent_list_2) )

sent_list_cut = ["/".join(rm_tokens(jieba.cut(part, cut_all=False))) for part in sent_list_2]

d = pd.DataFrame({"sentence":sent_list_2,"wordseg":sent_list_cut})
d.head()
d.to_csv("word_seg.csv",index = False,encoding = "gbk")



#导入数据
def loadDataset():
    '''导入文本数据集'''
    f = open('clean_text.txt','r')
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip())

    f.close()
    return dataset

text = loadDataset()


def word_count(texts):
#词频统计,转化成矩阵
    texts_list = [w for text_i in texts for w in text_i.split() ]
    word_count = np.array(Counter(texts_list).most_common())
    print (word_count[:10])
    csvfile = open("wordcount.csv",'w',newline='',encoding='utf-8-sig') 
    writer = csv.writer(csvfile)
    for row in word_count[0:1000,]:
        writer.writerow([row[0], row[1]])
    csvfile.close()

word_count(text)
# 统计2-gram词频,写入csv

def CountNgram(text,n=2,print_n=20):
    ngram_list = []
    for text_i in text:
        analyzer2 = ngrams(text_i.split(),n)
        Ngram_dict_i = Counter(analyzer2)        
        for k in Ngram_dict_i.keys():
            ngram_list.append("/".join(k))
    Ngram_dict = Counter(ngram_list)
    sortedNGrams = sorted(Ngram_dict.items(), key = operator.itemgetter(1), reverse=True) #=True 降序排列
    print("the top %d wordcount of %d gram model of period_1 is:\n" %(print_n,n),sortedNGrams[:print_n],"/n")

    csvfile = open("2gram_wordcount.csv",'w',newline='',encoding='utf-8-sig') 
    writer = csv.writer(csvfile)
    for line in sortedNGrams:
        writer.writerow([line[0],line[1]])
    csvfile.close()

CountNgram(text)
