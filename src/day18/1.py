# 8장. 텍스트빈도분석 - 2) 한글 단어 분석
## 한글 단어 분석을 위한 패키지 준비
import json
import re

from konlpy.tag import Okt # konlpy # C:\Program Files\Java\jdk-17\bin

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud

from konlpy.tag import Kkma
kkma = Kkma()	# 아마 설치가 잘 되지 않았다면 이 단계에서 에러가 났을 것이다.
print(kkma.nouns(u'안녕하세요 Soo입니다'))

# 1. 데이터 준비
### 1-1. 파일 읽기
inputFileName = 'etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명'
data = json.loads(open(inputFileName+'.json', 'r', encoding='utf-8').read())
print( data )     #작업 내용 확인용 출력

### 1-2. 분석할 데이터 추출
message = ''

for item in data:
    if 'message' in item.keys():
        message = message + re.sub(r'[^\w]', ' ', item['message']) + ''

print( message ) # 작업 내용 확인용 출력

### 1-3. 품사 태깅 : 명사 추출
nlp = Okt()
message_N = nlp.nouns(message)
print( message_N )   #작업 내용 확인용 출력

## 2. 데이터 탐색
### 2-1. 단어 빈도 탐색
count = Counter(message_N)
print( count )    #작업 내용 확인용 출력

word_count = dict()

for tag, counts in count.most_common(80):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))


### 히스토그램
font_path = "c:/Windows/fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname = font_path).get_name()
matplotlib.rc('font', family=font_name)

plt.figure(figsize=(12,5))
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation=75)

plt.show()



### 워드클라우드
wc = WordCloud(font_path, background_color='ivory', width=800, height=600).generate_from_frequencies(word_count)
plt.imshow(wc)
plt.show()
wc.to_file(inputFileName + '_cloud.jpg')


