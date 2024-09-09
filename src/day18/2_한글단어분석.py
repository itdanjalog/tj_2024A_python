# day18 -> 2_한글단어분석.py
# 자료 준비 : etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명.json


# 1. 데이터준비
# 1-1 파일 읽기
fileName = 'etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명.json'
file = open( fileName , 'r' , encoding='utf-8')
fileData = file.read( )
import json
# json.loads() : JSON형식 ---> PY형식 으로 변환
# json.dumps() : PY형식 ---> JSON형식 으로 변환
jsonData = json.loads( fileData )
# print( jsonData ) # 리스트 타입

# 1-2 분석 할 내용 추출
import re # 정규 표현식 모듈
message = ''
for item in jsonData :
    # 만약에 item(요소) 내 'message' key가 존재하면
    if 'message' in item.keys() :
        # print( item )
        # print( item['message'] ) # 분석할 문장
        # 전처리( 정규표현식 ) / ( 특수문자 제거 )
        # 분석할 문장내 정규표현식을 이용한 특수 문자를 제거 하고 공백 으로 치환 한다.
        message += re.sub( r'[^\w]' , ' ' , item['message'] )
        # print( message )
# print( message ) # 확인

# 1-3 품사 태깅 : 명사 추출
from konlpy.tag import Okt
okt = Okt() # 품사 태깅 객체 생성
words = okt.nouns( message ) # 품사(명사) 태깅 하기
#  print( words ) # 확인

# 2. 데이터 분석
# 2-1 단어 빈도 분석
from collections import Counter
wordsCount = Counter( words )
print( wordsCount ) #확인
# Counter( 리스트,튜플,문자열 ) #요소들의 빈도수 계산 객체

# 2-2 단어 빈도 (Counter)객체 를 딕셔너리 로 변환
word_count = { } # 빈 딕셔너리 생성  # = dict()
# word_count = dict()  vs  word_count = { }
for word , count in wordsCount.most_common( 80 ) :
    if len( word ) > 1 : # 단어의 길이가 1글자인 경우 제외
        word_count[word] = count  # { '단어' : 빈도수 }
print( word_count ) # 확인 # {'산업혁명': 22, '전자신문': 13, '산업': 10, '직업': 10, '기술': 8, }

# 3. 시각화1 : 히스토그램
import matplotlib.pyplot as plt
from matplotlib import font_manager , rc # 맷플롯립 에서 폰트 관련 객체
import matplotlib
# 맷플롯립 에서 한글화
# 3-1 시스템(컴퓨터) 폰트의 경로가 명확 하지 않을때
    # (1) 폰트 경로
font_path = "c:/windows/fonts/malgun.ttf" # 윈도우 기준의 폰트가 설치된 경로 : c:/windows/fonts/폰트명
    # (2) 폰트 설정 객체를 이용한 폰트 설정
font_name = font_manager.FontProperties( fname= font_path ).get_name() # 지정한 경로의 폰트 파일에서 폰트 이름을 가져옴
    # (3) 차트에 적용
matplotlib.rc( 'font' , family = font_name )

# 3-2 시스템(컴퓨터) 폰트의 경로가 명확 할때
# matplotlib.rc( 'font' , family = 'Malgun Gothic')

plt.bar( word_count.keys() , word_count.values() ) # plt.bar( x축값 , y축값 )
plt.xticks( rotation = 75 ) # x축 레이블 기울기
plt.show() # 차트 열기

# 4. 시각화2 : 워드클라우드
from wordcloud import WordCloud
wc = WordCloud( font_path , background_color='ivory' ).generate_from_frequencies( word_count )
# .generate_from_frequencies( ) : 단어와 빈도수를 딕셔너로 형식으로 받아서 시각화를 한다.
plt.imshow( wc ) # 맷플롭립 에 워드클라드 설정
plt.show( ) # 차트 열기






























