# day17 > 1_영문분석.py

import pandas as pd
import glob
import re
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS , WordCloud
# 최초 1번만 실행 하고 다운로드 후 주석처리
# import nltk
# nltk.download()

# [1] 여러개의 파일명을 불러오기 # glob.glob() 특정 패턴과 일치한 파일명을 모두 찾는 함수
all_files = glob.glob('myCabinetExcelData*.xls')
    # * : 와일드카드 : 모든 문자 대응  # myCabinetExcelData* : myCabinetExcelData시작하는 파일명 검색
# print( all_files ) # 확인후 주석

# [2] 여러개 파일명에 해당하는 엑셀파일을 호출해서 pd 로 변환하기 # 엑셀파일 --> pd
all_files_data = []
for file in all_files : # 모든 파일명을 하나씩 반복한다.
    # print( file )
    data_frame = pd.read_excel( file ) # 엑셀 모듈 : xlrd 모듈 관련 오류 발생 # xlrd 모듈 설치( 파일 -> 설정 -> 프로젝트(인터프리터) -> + -> 'xlrd' 검색후 설치  )
    # print(data_frame )
    all_files_data.append( data_frame ) # 불러온 엑셀 df 를 리스트에 담는다.

# print( all_files_data ) # 여러개 df 가 존재한다.

# [3] 데이터프레임 합치기 # .concat( 여러개프레임이저장된리스트 , axis= 0(세로)1(가로)   )
all_files_data_concat = pd.concat( all_files_data , axis= 0 , ignore_index = True  )
# print( all_files_data_concat )

# [4] 데이터프레임을 csv 로 변환/내보내기
all_files_data_concat.to_csv( 'riss_bigdata.csv' , encoding='utf-8' , index=False )

# - (프로젝트목표:학술문서의 제목 분석) 데이터 전처리
# [5] 데이터프레임의 제목(열) 만 추출
all_title = all_files_data_concat['제목']
# print( all_title )

# [6] 단어 토큰화 준비 ,
    # stopwords.words('english') : '영어' 불용어 목록 가져 오는 함수
    # WordNetLemmatizer() : 표제어 추출기 객체 생성
        # 표제어 : 단어의 원형(기본형) 찾는 과정 , running -> run , better -> good 변환 # 시제,단/복수,진행어 등등 일반화 과정
영문불용어목록 = stopwords.words('english')
# print( 영문불용어목록 )
표제어객체 = WordNetLemmatizer()

# [7] 단어 토큰화
words = [ ]
for title in all_title : # 제목 목록에서 제목 하나씩 반복하기
    # print( title )
    # 7-1 영문이 아닌것을 정규표현식을 이용해서 공백으로 치환
    EnWords = re.sub( r'[^a-zA-Z]+' , " " , str( title ) )
    # print( EnWords )

    # 7-2 소문자로 변환 하고 토큰화 #  word_tokenize( 문자열 ) # 지정한 문자열을 토큰(단어) 추출 하여 리스트로 반환
    EnWordsToken = word_tokenize( EnWords.lower() )
    # print( EnWordsToken )

    # 7-3 불용어 제거 ( 해당 토큰 리스트에 불용어가 포함되어 있으면 제외 ) # 반복문를 이용한 조건에 따른 새로운 리스트 생성
        # 리스트 컴프리헨션 사용X
    # EnWordsTokenStop = [] # 불용어가 제거된 토큰들 저장하는 리스트
    # for w in EnWordsToken :
    #     if w not in 영문불용어목록 :  # 해당 토큰(단어)가 불용어목록에 포함이 아니면
    #         EnWordsToken.append( w )

        # 리스트 컴프리헨션 사용O  ,
    EnWordsTokenStop = [ w for w in EnWordsToken if w not in 영문불용어목록 ]
        #   if 값 in 리스트 :  리스트내 해당 값이 존재하면 true 아니면 false
        #   if 값 not in 리스트 : 리스트내 해당 값이 존재하면 false 아니면 true

    # 7-4 표제어 추출 # 표제어객체.lemmatize( 단어 ) # 단어에서 시제, 단/복수 , 진행형 들을 일반화 단어로 추출
        # 리스트 컴프리헨션 사용0
    # EnWordsTokenStopLemma = [ 표제어객체.lemmatize( w ) for w in EnWordsTokenStop ]
        # 리스트 컴프리헨션 사용X
    EnWordsTokenStopLemma = []
    for w in EnWordsTokenStop :  # 불용어가 제거된 리스트(EnWordsTokenStop) 에서 표제어 추출
        EnWordsTokenStopLemma.append( 표제어객체.lemmatize( w ) )

    # print( EnWordsTokenStopLemma )
    # 제목을 정규화->토큰화->불용어제거->표제어추출 한 결과를 리스트에 담기
    words.append( EnWordsTokenStopLemma )
# 반복문 종료
# print( words )

# [8] 2차원 리스트를 1차원 변환 # reduce
    # reduce 사용O # reduce( 함수 , 2차원리스트 )
words2 = reduce( lambda x , y : x + y  , words )
# print( words2 )

# 파이썬 람다식 # lambda 매개변수1 , 매개변수2 : 실행문

    # 매개변수의 제곱을 하는 함수를 람다식 표현
# 제곱함수 = lambda x : x**2     # JS  x => x**2   # JAVA x -> x**2
# print( 제곱함수(1) ) # 1
# print( 제곱함수(2) ) # 4
#
#     # 람다식이 아닌 일반 함수
# def 제곱함수2( x ) :
#     return x ** 2
# print( 제곱함수2( 2 ) ) # 4

    # reduce 사용X
# words2 = []
# for w in words :
#     words2.extend( w ) # 리스트1.extend( 리스트2 ) # .extend() 두 리스트를 하나의 리스트로 반환 함수
# print( words2 )

# [9] 리스트내 요소 개수 세기 ( 단어 빈도 구하기 ) # Counter( 리스트 ) : 리스트내 요소들의 빈도 수 를 튜플로 해서 여러튜플 들을 리스트 반환 함수
count = Counter( words2 )
# print( count ) # Counter( { 단어 : 수 } , { 단어 : 수 } ) , { 단어 : 수 } )

# [10] 빈도가 높은것 만 추출 # 빈도가 높은 상위 50개 단어 추출
# print( count.most_common( 50 ) ) # [ ('단어', 수), ('단어', 수), ('단어', 수) ]
word_count = dict()  # dict()  빈 딕셔너리 생성
# for e in count.most_common(50) :
#     print( e ) # ('단어', 수 )
for ( tag , counts ) in count.most_common(50) :
    #print( tag )
    #print( counts )
    if( len(tag) > 1  ) : # 만약에 단어길이 가 1 초과 이면 # 단어가 1글자인 단어는 제외
        word_count[tag] = counts # 딕셔너리에 단어를 key로 하고 빈도수를 value 하여 쌍(엔트리) 저장
print( word_count )

# [11] 히스토그램
# plt.bar( x축값 , y축값 )
# plt.bar( [ 1 , 2 ,3 ] , [ 4,5, 6] )
    # 딕셔너리.keys() : 딕셔너리내 모든 key값 호출 반환 , 딕셔너리.keys() : 딕셔너리내 모든 value값 호출 반환 ,
# plt.bar( word_count.keys()  , word_count.values()  )
    # x축에 단어(keys)들 , y축 단어빈도수(values)들

# 딕셔너리 정렬 방법  # sorted( 딕셔너리 , key = 정렬기준 , reverse = True )
sorted_Keys = sorted( word_count , key = word_count.get , reverse=True )
    # key = word_count.get 는 get 메소드를 참조하여 각 키를 value(빈도수) 기준으로 정렬
    # reverse=True 는 내림차순 뜻 , 생략시 오름차순
# print( sorted_keys ) # ['data', 'big', 'analytics', 'analysis' ]

sorted_Values = sorted( word_count.values() , reverse=True )
# print( sorted_keys ) # [1645, 1354, 137, 67 ]

# print( range( len( word_count ) ) ) # range( 0 , 50 ) # 0 부터 49
plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation=85)

# 차트 실행
plt.show()

# [12] 결과 시각화
all_files_data_concat['doc_count'] = 0 # 데이터프레임의 필드(열) 추가
# print( all_files_data_concat['doc_count'] )
    # 출판일 별로 'doc_count' 의 기술통계
summary_year = all_files_data_concat.groupby( '출판일' , as_index=False )['doc_count'].count()
                #  데이터프레임 에서 '출판일' 열 기준으로 그룹화 하고
                # as_index=False 그룹화 할때 인덱스는 제외
                # .count() : 행 개수
# print( summary_year )
# plt.plot( [ 1, 2 , 3 ] , [ 4 , 5 , 6 ] )
# plt.plot( summary_year['출판일'] , summary_year['doc_count'] )
plt.grid( True )
plt.plot( range(len( summary_year) ) , summary_year['doc_count'] )
plt.xticks( range(len( summary_year) ) , [ text for text in summary_year['출판일'] ] )
# 차트 표시
plt.show()

# [13] 워드 클라우드
    # (1)문자열타입의 텍스트들의 워드 클라우드
# wc = WordCloud( ).generate("문자열타입")
    # (2)딕셔너리타입의 텍스트들의 워드 클라우드
# wc = WordCloud().generate_from_frequencies( "딕셔너리타입" )
wc = WordCloud().generate_from_frequencies( word_count )
plt.imshow( wc )
plt.show()

# 워드 클라우드 결과 이미지 저장
wc.to_file( "wordCloud.jpg")
























