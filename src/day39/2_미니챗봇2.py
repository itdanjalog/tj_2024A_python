# 2_미니챗봇.py
import numpy as np
import pandas as pd

# RNN 기본구조 : 1. 데이터수집 2.전처리 3.토큰화/패딩 4. 모델구축 5.모델학습 6.모델평가(튜닝) 7.모델예측

# 1. 데이터 수집 # csv , db , 함수(코드/메모리)
def func1( x ) :
    return f"{x}의 주문 했습니다."

data = {
    "안녕하세요": "안녕하세요! 무엇을 도와드릴까요?",
    "오늘 날씨 어때요?": "오늘은 맑고 화창한 날씨입니다.",
    "지금 몇 시에요?": "현재 시간은 오후 3시입니다.",
    "좋은 책 추천해 주세요": "최근에 인기가 많은 책은 '파이썬 데이터 분석'입니다.",
    "고마워요": "천만에요! 더 필요한 것이 있으면 말씀해주세요.",
    "콜라 주문 요청 합니다.": func1( "콜라" ) ,
    "사이다 주문 요청 합니다.": func1( "사이다" )
}
# 2. 데이터 전처리
inputs = list( data.keys() ) # 질문
outputs = list( data.values() ) # 응답
from konlpy.tag import Okt
import re # 정규표현식
okt = Okt()
def preprocess( text ) :
    # 한글 과 띄어쓰기(\s) 를 제외한 문자 제거 # ^부정
    result = re.sub( r'[^가-힣\s]' , '' , text ) # 정규표현식 # 일반적인 문자열 정규표현식
    print( result )
    # 2. 형태소 분석
    result = okt.pos( result ); print( result ) # [ ('라면','Noun') , ('하다',Verb) ]
    # 3. 명사(Noun)와 동사(Verb) 와 형용사(Adjective) 외 제거
    # 형태소 분석기가 각 형태소들을 명칭하는 단어들 (pos)변수 존대한다.
    result = [ word for word , pos in result if pos in ['Noun','Verb','Adjective'] ]
    # 4. 불용어 생략
    # 불용어 목록
    stopwords = ['은', '는', '이', '가', '을', '를', '에', '의', '도', '와', '한', '하다', '고', '들']
    # 불용어 제거
    result = [word for word in result if word not in stopwords]
    # 5. 반환
    return " ".join( result ).strip( ) # strip() 앞뒤 공백 제거 함수
# 전처리 실행  # 모든 질문을 전처리 해서 새로운 리스트
processed_inputs = [ preprocess(질문) for 질문 in inputs ]
print( processed_inputs )
# ['안녕하세요', '오늘 날씨 어때요', '지금 몇 시', '좋은 책 추천 해 주세요', '고마워요']

# 3. 토크나이저
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts( processed_inputs ) # 전처리된 단어 목록을 단어사전 생성
print( tokenizer.word_index ) # 사전확인
# {'안녕하세요': 1, '오늘': 2, '날씨': 3, '어때요': 4, '지금': 5, '몇': 6, '시': 7, '좋은': 8, '책': 9 } ~~

input_sequences = tokenizer.texts_to_sequences( processed_inputs ) # 벡터화
print( input_sequences )
# ['안녕하세요', '오늘 날씨 어때요', '지금 몇 시', '좋은 책 추천 해 주세요', '고마워요']
# [[1], [2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13]]

max_sequence_length = max( len(문장) for 문장 in input_sequences ) # 여러 문장중에 가장 긴 단어의 개수
print( max_sequence_length ) # '좋은 책 추천 해 주세요' # 5

input_sequences = pad_sequences( input_sequences  , maxlen=max_sequence_length ) # 패딩화 # 가장 길이가 긴 문장 기준으로 0으로 채우기
print( input_sequences ) #  '오늘 날씨 어때요' --> [ 2  3  4 ] --> [ 0 0 2 3 4 ] # 좋은 성능을 만들기 위해 차원을 통일

# 종속변수 # 데이터프레임 --> 일반 배열 변환
# output_sequences = np.array(  outputs  )
# print( output_sequences )
output_sequences = np.array( range( len( outputs ) ) )
print( output_sequences )

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , LSTM , Dense , Bidirectional , Dropout , BatchNormalization#
# 1. 모델
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(512, return_sequences=True)))  # LSTM 유닛 수 증가
model.add(Dropout(0.5))  # 드롭아웃 비율
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(0.5))  # 추가 드롭아웃
model.add(Dense(len(outputs), activation='softmax'))
# 2. 컴파일
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 학습률 조정
model.compile( loss='sparse_categorical_crossentropy' , optimizer=optimizer , metrics=['accuracy'] )
# 3. 학습
model.fit( input_sequences , output_sequences  , epochs=50, batch_size=32 )
# 4. 예측하기
def response( text ) :
    text = preprocess( text )# 1. 예측할 값도 전처리 한다.
    text = tokenizer.texts_to_sequences( [ text ] )  # 2. 예측할 값도 토큰 과 패딩  # 학습된 모델과 데이터 동일
    text = pad_sequences( text , maxlen= max_sequence_length )
    result = model.predict( text ) # 3. 예측
    print( result )
    max_index = np.argmax( result )  # 4. 결과 # 가장 높은 확률의 인덱스 찾기
    result = outputs[max_index]  # 5.
    if callable(result):
        return result()
    else :
        return result

# 확인
print( response('안녕하세요') ) # 질문이 '안녕하세요' , 학습된 질문 목록중에 가장 높은 예측비율이 높은 질문의 응답을 출력한다.
# 서비스 제공한다. # 플라스크
while True :
    text = input( '사용자 : ' ) # 챗봇에게 전달할 내용 입력받기
    result = response( text ) # 입력받은 내용을 함수에 넣어 응답을 예측를 한다.
    print( f'챗봇 : { result }') # 예측한 응답 출력한다.