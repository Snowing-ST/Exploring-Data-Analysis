# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:21:59 2018

@author: situ
"""

import requests,re,time
import os
import pandas as pd
import numpy as np
from urllib.parse import urlencode
from lxml import etree


headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
}

replace_dict={
    "&#xf09f":"0",
    "&#xeff8":"1",
    "&#xecfa":"2",
    "&#xf748":"3",
    "&#xf298":"4",
    "&#xed58":"5",
    "&#xee56":"6",
    "&#xe253":"7",
    "&#xe504":"8",
    "&#xecfd":"9"}
def get_links(start_url,n,replace_dict):
    all_pd = pd.DataFrame()
    for i in list(range(1,n+1)):
        print("————————————正在爬取第%d页招聘信息———————————————"%i)
        url = start_url+"&p=%s"%str(i)
        try:
            wb_data = requests.get(url,headers=headers)
            wb_data.encoding=wb_data.apparent_encoding
            links = re.findall('class="name-box clearfix".*?href="(.*?)"',wb_data.text,re.S)
            for link in links:
                print(link)
                try:
                    one_pd = get_infos('https://www.shixiseng.com'+link,replace_dict)
                except:
                    one_pd = pd.DataFrame({"url":link,"jobname":"","salary":"","address":"",
                    "education":"","jobway":"","month":"",
                    "jobgood":"","contents":"","compname":"",
                    "city":"","size":"","industry":""})
                    print("can't crawl"+link)
                all_pd = all_pd.append(one_pd)
        except:
            print("can't reach page %d"%i)
            pass
                
    return all_pd
        
def get_infos(url,replace_dict):
    one_dict = {}
    wb_data = requests.get(url,headers=headers)
    print(wb_data.status_code)
    wb_data.encoding=wb_data.apparent_encoding
    jobname = re.findall('<div class="new_job_name" title="(.*?)">',wb_data.text,re.S)
    salarys = re.findall('class="job_money cutom_font">(.*?)</span>',wb_data.text,re.S)
    addresses = re.findall('class="job_position">(.*?)</span>',wb_data.text,re.S)
    educations = re.findall('class="job_academic">(.*?)</span>',wb_data.text,re.S)
    jobways = re.findall('class="job_week cutom_font">(.*?)</span>',wb_data.text,re.S)
    months = re.findall('class="job_time cutom_font">(.*?)</span>',wb_data.text,re.S)
    jobgoods = re.findall('class="job_good".*?>(.*?)</div>',wb_data.text,re.S)
    contents = re.findall(r'div class="job_til">([\s\S]*?)<div class="job_til">', wb_data.text, re.S)[0].replace(' ','').replace('\n', '').replace('&nbsp;', '')
    contents = re.sub(r'<[\s\S]*?>', "", str(contents))
    compname = re.findall('class="job_com_name">(.*?)</div>',wb_data.text,re.S)
    compintro = re.findall('<div class="job_detail job_detail_msg"><span>([\s\S]*?)</span></div>',wb_data.text,re.S)
    city,size,industry = re.sub(r'<[\s\S]*?>', " ", str(compintro[0])).split()
    for salary,address,education,jobway,month,jobgood in zip(salarys,addresses,educations,jobways,months,jobgoods):
        for key, vaule in replace_dict.items():
            salary = salary.replace(key, vaule)
            jobway = jobway.replace(key,vaule)
            month = month.replace(key,vaule)
            one_dict = {"url":url,"jobname":jobname,"salary":salary,"address":address,
                    "education":education,"jobway":jobway,"month":month,
                    "jobgood":jobgood,"contents":contents,"compname":compname,
                    "city":city,"size":size,"industry":industry}
#    list_i=[url,salary,address,education,jobway,month,jobgood,contents,compname,city,size,industry]
    print(jobname)
    one_pd = pd.DataFrame(one_dict)
    return one_pd
    
    
if __name__ == '__main__':
    os.chdir("E:/graduate/class/EDA/final")
    print('请输入您想爬取内容的关键字：')
    compRawStr = input('关键字： \n')     #键盘读入 多个关键字则用空格隔开
    print('正在爬取“' + compRawStr.capitalize()+ '”有关实习信息!')
    d = {'k': compRawStr.encode('utf-8')}
    word = urlencode(d)

    start_url = "https://www.shixiseng.com/interns/st-intern_c-None_?%s" %word
    result = requests.get(start_url,headers=headers)
#    result.status_code
    result.encoding = 'utf-8'
    selector = etree.HTML(result.text)  
    last_page_link = selector.xpath('//*[@id="pagebar"]/ul/li[10]/a/@href')
    n = int(last_page_link[0].split("p=")[1])
    print("将爬取%d页的招聘信息"%n)
    time_start=time.time()
    df = get_links(start_url,n,replace_dict)
    df.to_csv(compRawStr+"_共"+str(n)+"页.csv",index = False,encoding = "gb18030")
    time_end=time.time()
    print("成功爬取%d条关于【%s】的招聘信息"%(len(df),compRawStr))
    print('totally cost %f seconds'%(time_end-time_start))


