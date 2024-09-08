# 8장. 텍스트빈도분석 - 2) 한글 단어 분석
'''
**품사 태깅(Part-of-Speech Tagging, POS 태깅)**이란,
문장에서 각 단어가 어떤 품사에 속하는지를 자동으로 구분하여 태그를 붙이는 작업입니다.
품사는 명사, 동사, 형용사, 부사, 조사 등과 같이 단어의 문법적 역할을 나타내며,
품사 태깅은 텍스트 분석에서 매우 중요한 자연어 처리(NLP) 기술 중 하나입니다.

품사(Part of Speech): 단어의 문법적 역할을 나타내는 범주입니다. 예를 들어:
명사(N): 사물이나 사람의 이름 (예: '사과', '책')
동사(V): 행동이나 상태를 나타내는 말 (예: '먹다', '가다')
형용사(A): 사물의 성질이나 상태를 나타내는 말 (예: '크다', '예쁘다')
부사(Adv): 동사나 형용사를 수식하는 말 (예: '빠르게', '정말')
태그(Tag): 각 단어에 부착된 품사 정보입니다. 예를 들어, "사과를 먹다"라는 문장에서 "사과"는 명사(N), "먹다"는 동사(V)로 태깅됩니다.
'''

from konlpy.tag import Okt

text = "나는 사과를 먹었다."
okt = Okt()
tagged_words = okt.pos(text) # nouns 명사만.
print(tagged_words)
'''
'나': 명사 (Noun)
'는': 조사 (Josa)
'사과': 명사 (Noun)
'먹었다': 동사 (Verb)
'''

from konlpy.tag import Okt

# 경제 기사 예제 텍스트
text = """
한국은행은 최근 금리 동결을 발표했다. 이로 인해 경제 성장률에 미치는 영향에 대한 우려가 커지고 있다. 
전문가들은 글로벌 경제 불확실성, 수출 감소, 내수 위축 등을 고려할 때 금리 인상보다는 동결이 적절하다고 평가했다. 
금융 시장에서는 금리 동결 소식에 긍정적인 반응을 보이고 있다.
"""

# Okt 형태소 분석기 인스턴스 생성
okt = Okt()

# 품사 태깅 수행
tagged_words = okt.pos(text)

# 결과 출력
for word, tag in tagged_words:
    print(f'{word}: {tag}')





## 한글 단어 분석을 위한 패키지 준비
# import json
# import re
# from konlpy.tag import Okt
# from collections import Counter
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# from wordcloud import WordCloud
# ======================================== ========================================
# 1. 데이터 준비
### 1-1. 파일 읽기
import json
inputFileName = 'etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명'
file = open(inputFileName+'.json', 'r', encoding='utf-8')
fileData = file.read()
data = json.loads( fileData )
print( data )     #작업 내용 확인용 출력
# ======================================== ========================================
### 1-2. 분석할 데이터 추출
import re
message = ''    # # 빈 문자열 'message'를 초기화 (추출한 메시지를 저장할 공간)
for item in data: # 'data' 리스트 내의 각 요소(딕셔너리)를 하나씩 반복
    if 'message' in item.keys(): # 각 딕셔너리에서 'message'라는 키가 있는지 확인
        # 'message' 값을 가져와서 정규 표현식으로 특수 문자를 제거하고, 공백으로 치환하여 message에 추가
        message = message + re.sub(r'[^\w]', ' ', item['message']) + ''
print( message ) # 작업 내용 확인용 출력
# ======================================== ========================================

### 1-3. 품사 태깅 : 명사 추출
from konlpy.tag import Okt
nlp = Okt() #  # 'Okt' 클래스의 인스턴스를 생성. 이 클래스는 한국어 형태소 분석을 위한 도구
'''
Okt(): KoNLPy 라이브러리에서 제공하는 형태소 분석기 중 하나입니다. 
Okt는 Twitter 형태소 분석기로도 알려져 있으며, 한국어 텍스트에서 단어를 분석하거나 태깅할 때 유용합니다.
'''
message_N = nlp.nouns(message) ## 'message' 문자열에서 명사만 추출하여 리스트로 반환
# Okt 객체의 메서드로, 입력된 텍스트에서 명사만을 추출해 리스트 형태로 반환합니다.

print( message_N )   #작업 내용 확인용 출력
# ======================================== ========================================

## 2. 데이터 탐색
### 2-1. 단어 빈도 탐색
from collections import Counter
count = Counter(message_N ) # 'message_N' 리스트에 있는 명사들의 빈도수를 계산하여 'count'에 저장
'''
Python의 collections 모듈에 있는 클래스입니다. 리스트, 튜플, 문자열 등의 데이터에서 요소들의 빈도수를 계산하고, 
이를 딕셔너리 형태로 반환합니다. 각 요소(여기서는 명사)가 몇 번 등장했는지를 보여줍니다.
'''
print( count )    #작업 내용 확인용 출력
# ======================================== ========================================

# word_count = dict() # 더 명시적이고, 객체 지향적인 방식입니다.
word_count = { } # 코드가 간결하고 빠릅니다. 일반적으로 빈 딕셔너리를 초기화할 때 가장 많이 사용됩니다.
# 일반적으로 빈 딕셔너리를 만들 때는 {}를 많이 사용하지만, 상황에 따라 dict()도 유용할 수 있습니다.

for tag, counts in count.most_common(80): # 명사 빈도수를 내림차순으로 정렬한 후 상위 80개의 (단어, 빈도수) 쌍을 하나씩 가져옴
    if(len(str(tag))>1): # 단어의 길이가 1글자보다 큰 경우만 선택 (1글자는 제외)
        word_count[tag] = counts # 'tag'를 키로, 'counts'를 값으로 'word_count' 딕셔너리에 추가
        print("%s : %d" % (tag, counts))  # 단어(tag)와 빈도수(counts)를 출력

# ======================================== ========================================

### 히스토그램
from matplotlib import font_manager, rc
import matplotlib
import matplotlib.pyplot as plt

font_path = "c:/Windows/fonts/malgun.ttf" # 말굽 폰트 파일 경로 지정 (한글 텍스트를 시각화할 때 사용)
# c:/Windows/fonts

font_name = font_manager.FontProperties(fname = font_path).get_name() #  # 지정한 폰트 파일에서 폰트 이름을 가져옴
matplotlib.rc('font', family=font_name) # matplotlib에 폰트 설정 (한글 폰트를 사용하도록 설정)

plt.figure(figsize=(12,5)) # 그래프의 크기를 가로 12인치, 세로 5인치로 설정
plt.xlabel('키워드') # x축 레이블을 '키워드'로 설정
plt.ylabel('빈도수')  # y축 레이블을 '빈도수'로 설정
plt.grid(True) # 그래프에 그리드 추가

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)  # word_count 딕셔너리에서 값을 기준으로 내림차순으로 정렬된 키(단어) 목록 생성
sorted_Values = sorted(word_count.values(), reverse=True) # word_count의 값을 내림차순으로 정렬하여 빈도수 목록 생성

plt.bar(range(len(word_count)), sorted_Values, align='center')  # 막대그래프를 그리며, 각 막대의 높이는 빈도수로 설정
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation=75)# x축에 단어를 표시하며, 단어의 각도는 75도로 회전시켜 보기 쉽게 설정

plt.show()  # 설정한 그래프를 화면에 출력
# ======================================== ========================================


### 워드클라우드

from wordcloud import WordCloud

#wc = WordCloud(font_path, background_color='ivory', width=800, height=600).generate_from_frequencies(word_count)

wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
'''
font_path: 워드 클라우드에 한글이 포함될 경우, 한글이 깨지지 않도록 앞에서 설정한 폰트 경로(예: malgun.ttf)를 사용합니다.
background_color='ivory': 워드 클라우드의 배경색을 'ivory'로 설정합니다.
width=800, height=600: 워드 클라우드의 가로 크기와 세로 크기를 각각 800픽셀과 600픽셀로 설정합니다.
'''
cloud=wc.generate_from_frequencies(word_count)
#  단어와 그 단어가 등장한 빈도수를 딕셔너리 형태로 받아, 해당 빈도에 따라 단어의 크기를 다르게 시각화합니다

plt.figure(figsize=(8,8)) # 워드 클라우드 이미지를 표시할 그래프의 크기를 8x8인치로 설정
plt.imshow(cloud)  # 생성된 워드 클라우드 이미지를 그래프에 표시
plt.axis('off')  # 축을 표시하지 않도록 설정 (워드 클라우드에서는 축이 필요 없기 때문)
plt.show() # 워드 클라우드 이미지를 화면에 출력

cloud.to_file(inputFileName + '_cloud.jpg') # 생성된 워드 클라우드 이미지를 파일로 저장합니다.


