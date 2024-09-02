import nltk
import pandas as pd
import glob
import re
from functools import reduce

from nltk.tokenize import word_tokenize # nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# nltk.download() # 한번만 실행

all_files = glob.glob('myCabinetExcelData*.xls')

print( all_files ) #출력하여 내용 확인

all_files_data = [] #저장할 리스트

for file in all_files:
    data_frame = pd.read_excel(file) # imstall xlrd
    all_files_data.append(data_frame)

print( all_files_data[0]  )  #작업 내용 확인용 출력


all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)

print( all_files_data_concat ) #작업 내용 확인용 출력


all_files_data_concat.to_csv('riss_bigdata.csv', encoding='utf-8', index = False)


# 제목 추출
all_title = all_files_data_concat['제목']
print( all_title ) #작업 내용 확인용 출력


stopWords = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

words = []

for title in all_title:
    EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title))
    EnWordsToken = word_tokenize(EnWords.lower())
    EnWordsTokenStop = [w for w in EnWordsToken if w not in stopWords]
    EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop]
    words.append(EnWordsTokenStopLemma)

print(words)  #작업 내용 확인용 출력