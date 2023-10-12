# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:10:12 2023

@author: Murahari Chavali
"""
import os
os.environ["OMP_NUM_THREADS"] = '1'

import os
import pandas as pd
import numpy as np
import bs4 as bs
import urllib.request
import re
import spacy 
import string,unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer,WordNetLemmatizer

lem=WordNetLemmatizer()


os.chdir(r"C:\Users\Murahari Chavali\Desktop\colab\Colab Notebooks\week10\sep\25th,26th, - webscrapping\25th,26th, - webscrapping\xml_many articles")

from glob import glob

path=r"C:\Users\Murahari Chavali\Desktop\colab\Colab Notebooks\week10\sep\25th,26th, - webscrapping\25th,26th, - webscrapping\xml_many articles"
all_files=glob(os.path.join(path,'*.xml'))


import xml.etree.ElementTree as ET

dfs=[]
for i in all_files:
    tree=ET.parse(i)
    root=tree.getroot()
    root=ET.tostring(root,"utf8").decode('utf8')
    dfs.append(root)

test=dfs[0]
import bs4 as bs
import urllib.request
import re

parsed_article = bs.BeautifulSoup(dfs[1],'xml')
parageaphs=parsed_article.find_all('para')

for element in parsed_article.find_all():
    print(element)


article_text_full=''

for p in parageaphs:
    article_text_full=article_text_full+p.text
    print(p.text)

#==============================================
def data_preprocessing (each_value):
    parsed_article=bs.BeautifulSoup(each_value,'xml')
    
    parageaph=parsed_article.find_all('para')
    
    article_text_full=' '
    
    for i in parageaph:
        article_text_full += i.text
        
    return article_text_full



data=[data_preprocessing (i) for i in dfs]


from bs4 import BeautifulSoup
soup=BeautifulSoup(dfs[0],'html.parser')


print(soup.prettify())

def remove_stop_words(each_item):
    nlp=spacy.load('en_core_web_sm')
    
    punctuation= string.punctuation
    
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')
    Symbols=" ".join(string.punctuation).split(" ")+["-", "...", "”", "”"]
    
    stopwords=nltk.corpus.stopwords.words('english')+Symbols
    
    
    doc=nlp(each_item,disable=["parser",'ner'])
    tokens=[tok.lemma_.lower().strip() for tok in doc if tok.lemma_ !='-PRON-']
    tokens=[tok for tok in tokens if tok not in stopwords and tok not in punctuation]
    s=[lem.lemmatize(word) for word in tokens]
    tokens=' '.join(s)
    
    
    
    article_text=re.sub(r'\[[0-9]*\]]'," ",tokens)
    article_text=re.sub(r'\s+'," ",article_text)
    
    
    formatted_article_text=re.sub('[^a-zA-Z]', ' ', article_text)
    
    formatted_article_text=re.sub(r'\s+',' ',formatted_article_text)
    formatted_article_text=re.sub(r'\W*\b\w{1,3}\b','',formatted_article_text)
    
    return formatted_article_text
    
    
    
clean_data=[remove_stop_words(file) for file in data]


all_words=' '.join(clean_data)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud=WordCloud(width=400, height=200, random_state=21, max_font_size=110).generate(all_words)
    
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
    
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
freq=word_tokenize(all_words)
freqdist=FreqDist(freq)
freqdist.plot(100)
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
    
    
vectorirzer=CountVectorizer(stop_words=stopwords.words('english')).fit(clean_data)   

vectorirzer.get_feature_names_out()
x=vectorirzer.transform(clean_data).toarray() 


data_final=pd.DataFrame(x,columns=vectorirzer.get_feature_names_out())
    
    
from sklearn.feature_extraction.text import TfidfTransformer
tfid=TfidfTransformer().fit(data_final.values)
    
x=tfid.transform(x).toarray()
x=normalize(x)
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=2).fit(x)
    
    
    
kmeans.predict(x)
    


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("THIS IS GRAPH FOR ELBOW METHOD")
plt.xlabel('Number of clusters')
plt.ylabel("WCSS")
plt.savefig('elbow.png')
plt.show()
    
"""

from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
"""
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

distortions=[]

inertias=[]
mapping1 ={}
mapping2={}
K=range(1,15)
for i in K:
    # Building and fitting the model
    kmeansmodel=KMeans(n_clusters=i).fit(x)
    kmeansmodel.fit(x)
    
    distortions.append(sum(np.min(cdist(x,kmeansmodel.cluster_centers_,'euclidean'),axis=1))/x.shape[0])
    
    inertias.append(kmeansmodel.inertia_)
    
    mapping1[i]=sum(np.min(cdist(x,kmeansmodel.cluster_centers_,'euclidean'),axis=1))/x.shape[0]
    
    mapping2[i]=kmeansmodel.inertia_
    

    
    
for key,val in mapping1.items():
    print(str(key)+'  :  '+str(val))

plt.plot(K,distortions,'s-.m')

plt.xlabel("Value of k")
plt.ylabel("Distortions")
plt.title("Elbow graph")


plt.show()



for k,v in mapping2.items():
    print(str(k),"=",str(v))
plt.plot(K,inertias,'2-y')
plt.xlabel("Value of K")
plt.ylabel("Inertia")
plt.title("The elbow method using Inertia")
plt.show()


true_k=6
model=KMeans(n_clusters=true_k,init='k-means++',max_iter=100,n_init=1)
model.fit(x)


import warnings
warnings.filterwarnings('ignore')



order_centeriod=model.cluster_centers_.argsort()[:,::-1]
terms=vectorirzer.get_feature_names_out()


for i in range(true_k):
    print('cluster %d' % i),
    for ind in order_centeriod[i,:50]:
        print("%s"%terms[ind])
        
#==================================
#cleaning completed NEXT grouping

cluste_result=pd.DataFrame(clean_data,columns=['text'])

cluste_result['group']=model.predict(x)

from wordcloud import WordCloud

normal_words=''.join([text for text in cluste_result.loc[cluste_result['group']==0,'text']])

wordcloud=WordCloud(width=800,height=600,random_state=21,max_font_size=110).generate(normal_words)
        
plt.figure(figsize=(10,7))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')
plt.show()
        
        
        
for i in range(0,6):
    normal_words=''.join([text for text in cluste_result.loc[cluste_result['group']==i,'text']])
    wordcloud=WordCloud(width=800,height=700,random_state=21,max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis=('off')
    plt.title(str(i))
    plt.show()
        
        
        
        
        
        
        
        
        
        
        def token(sentance):
            tok=sentance.split()
            
            return tok
             
            
            

        cluster_result['words'] = [token(sentance) for sentance in cluster_result['text']]
        
        
        
        
        