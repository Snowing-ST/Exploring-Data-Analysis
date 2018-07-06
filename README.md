# Exploring-Data-Analysis
- 探索性数据分析课程论文

**[基于文本聚类的招聘信息技能要求提取与量化.pdf](https://github.com/Snowing-ST/Exploring-Data-Analysis/blob/master/%E5%9F%BA%E4%BA%8E%E6%96%87%E6%9C%AC%E8%81%9A%E7%B1%BB%E7%9A%84%E6%8B%9B%E8%81%98%E4%BF%A1%E6%81%AF%E6%8A%80%E8%83%BD%E8%A6%81%E6%B1%82%E6%8F%90%E5%8F%96%E4%B8%8E%E9%87%8F%E5%8C%96.pdf)**

本文通过爬取实习僧网站“数据分析”一职的实习信息，对“职位描述”的文本进行预处理、分句，使用文本聚类的方式提取每条实习信息中其中的描述专业技能的句子，并对其描述的专业技能进行量化，从而探究专业技能对薪资的影响。本文所述的方法还可用于提取其他岗位、其他要求等，为大学生提供最直接、最真实的岗位信息，从而使他们对感兴趣的职业有所了解，对他们的学习方向提供建议，使其和能更明确地为求职作准备。

1. [实习僧爬虫:crawl_shixiseng.py](https://github.com/Snowing-ST/Exploring-Data-Analysis/blob/master/crawl_shixiseng.py)
2. [文本预处理:text_preprocess.py](https://github.com/Snowing-ST/Exploring-Data-Analysis/blob/master/text_preprocess.py)
3. [文本聚类提取并量化职位描述中的技能信息:text_cluster.py](https://github.com/Snowing-ST/Exploring-Data-Analysis/blob/master/text_cluster.py)
    - kmeans
    - GMM
    - NMF
4. [薪资与技能的回归分析:analysis.py](https://github.com/Snowing-ST/Exploring-Data-Analysis/blob/master/analysis.py)

![词云图](https://github.com/Snowing-ST/Exploring-Data-Analysis/blob/master/tagxedo.png)
![薪资与技能](https://github.com/Snowing-ST/Exploring-Data-Analysis/blob/master/salary_and_skill.png)