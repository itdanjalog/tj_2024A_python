## - 자연어처리패키지 nltk download
#### - 아나콘다에 nltk 가 기본으로 설치되어있으므로, pip으로 설치할 필요없음.
####    하지만, 최초 한번은 nltk의 리소스를 다운로드 받아야함.
import nltk     # nltk.download() 를 하기위해, import 함.
#nltk.download()  # 최초 한번만 설치: download 창이 뜨면, 모두 선택하고 [Download] 버튼 클릭!

# 1. 데이터 준비
import pandas as pd
import glob
import re
from functools import reduce

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

### 1-1. 파일 병합
all_files = glob.glob('myCabinetExcelData*.xls')
print( all_files )  #출력하여 내용 확인


all_files_data = [] #저장할 리스트

for file in all_files:
    data_frame = pd.read_excel(file) # xlrd
    all_files_data.append(data_frame)

print(  all_files_data[0])      #작업 내용 확인용 출력

all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)

print(  all_files_data_concat)    #작업 내용 확인용 출력

all_files_data_concat.to_csv('riss_bigdata.csv', encoding='utf-8', index = False)

### 1-2. 데이터 전처리 (Pre-processing)
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

words2 = list(reduce(lambda x, y: x+y,words))
print(words2)  #작업 내용 확인용 출력
##################################################################################################################
# 2. 데이터 탐색 및 분석 모델 구축
## 2-1. 단어 빈도 탐색
count = Counter(words2)
print( count )   #작업 내용 확인용 출력

word_count = dict()

for tag, counts in count.most_common(50):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))

## 2-2 단어 빈도 히스토그램
# 히스토그램 표시 옵션
plt.figure(figsize=(12,5))
plt.xlabel("word")
plt.ylabel("count")
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation=85)

plt.show()

##################################################################################################################
# 3. 결과 시각화
## 3-1. 연도별 데이터 수
all_files_data_concat['doc_count'] = 0
summary_year = all_files_data_concat.groupby('출판일', as_index=False)['doc_count'].count()
summary_year  #출력하여 내용 확인

plt.figure(figsize=(12,5))
plt.xlabel("year")
plt.ylabel("doc-count")
plt.grid(True)

plt.plot(range(len(summary_year)), summary_year['doc_count'])
plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])

plt.show()

## 3-2. 워드클라우드
stopwords=set(STOPWORDS)
wc=WordCloud(background_color='ivory', stopwords=stopwords, width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()

#### - 워드 클라우드에 나타나는 단어의 위치는 실행 할 때마다 달라진다. ☺
cloud.to_file("riss_bigdata_wordCloud.jpg")

## 여기서 잠깐!! :
### 검색어로 사용한 big'과 'data' 빈도가 압도적으로 많으므로, 정확한 단어 빈도 분석을 위해 검색어를 제거하고 단어 빈도 히스토그램을 다시 그려보자.
#검색어로 사용한 'big'과 'data' 항목 제거 하기
del word_count['big']
del word_count['data']

# 히스토그램 표시 옵션
plt.figure(figsize=(12,5))
plt.xlabel("word")
plt.ylabel("count")
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation=85)

plt.show()
















