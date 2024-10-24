# day40 --> 1_자연어생성 p.310

import tensorflow as tf
import pandas as pd
from torch.backends.mkl import verbose

# 1. 데이터 수집 # .get_file( )
file = tf.keras.utils.get_file(
    'ratings_train.txt' , # 파일명
    origin = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',# 다운로드 받을 링크
    extract = True # 압축 설정
)
df = pd.read_csv( file , sep = '\t' )
print( df[1000:1007] ) # 데이터 중 임의의 행 확인
# 2. 데이터 전처리
from konlpy.tag import Okt # 한글 형태소 분석기 클래스 # Open Korean Text
okt = Okt() # 파이썬객체생성 방법 : 클래스명() # vs Okt okt = new Okt()
def word_tokenization( text ) :
    # 방법1
    list = [ ]
    result = okt.morphs( text ) # okt.morphs( ) :  vs okt.pos( ) :
    for word in result :
        list.append( word )
    return list
    # 방법2
    # return  [ word for word in okt.morphs( text ) ] # 리스트 컴프리헨션
def preprocessing( df ) :
    df = df.dropna( ) # 데이터프레임(df) # 결측값 제거
    df = df[ 1000:2000 ] # 샘플 데이터 1000개 사용
    df['document'] = df['document'].str.replace( "[^A-Za-z0-9가-힣-ㄱ-ㅎㅏ-ㅣ ]", "" , regex = True ) # 데이터프레임(df)['열/속성 이름'] # 데이터프레임(df) 안에서의
    # "안녕하세요".replace( )    : 문자열(String)을 치환하는 함수
    # df['document'].replace( ) : 특정 열/속성의 여러개 문자열 치환 함수 # str : 문자열 반환 # 정규표현식 사용
    # * 서로 다른 객체들이 동일한 이름의 함수/기능 제공하는 경우  # 매개변수와 반환이 다를수 있다.
    data = df['document'].apply( (lambda x : word_tokenization( x ) ) )
    # data = df['document'].apply( (lambda x : [ word for word in okt.morphs( x ) ] ) )
    return data
review = preprocessing( df )
print( review )
print( review[ : 10 ] )

# 3. 토큰화
from tensorflow.keras.preprocessing.text import Tokenizer # 토큰 관련 클래스
from tensorflow.keras.preprocessing.sequence import pad_sequences # 패딩 관련 클래스
tokenizer = Tokenizer() # 객체 생성

# 1. 정말 최고의 명작 성인이 되고 본 이집트의 왕자는 또 다른 감동 그자체네요
# 2. [14, 31, 3, 140, 540, 1, 412, 63, 1382, 3, 792, 10, 128, 180, 46, 61, 239, 90, 57]
# 3. 시퀀스
def get_tokens( review ) :
    # 토큰객체.fit_on_texts( ) : 각 단어의 인덱스(숫자)를 대응하는 *단어사전* 생성 # 빈도수에 따라 인덱스(숫자) 위치가 결정된다.
    tokenizer.fit_on_texts( review )
    print( tokenizer.word_index )
    total_words = len( tokenizer.word_index ) + 1
    print( total_words )
    # 각 문장을 숫자(벡터)로 변환 # .texts_to_sequences( )
    tokenized_sentences = tokenizer.texts_to_sequences( review ) # 위의 정의된 단어사전 기준으로 단어를 인덱스(숫자) 변환
    print( tokenized_sentences  ) # [14, 31, 3, 140, 540, 1, 412, 63, 1382, 3, 792, 10, 128, 180, 46, 61, 239, 90, 57]

    input_sequences = [ ]
    for token in tokenized_sentences : # 문장( 여러 벡터 ) 를 하나씩 반복
        for t in range( 1 , len( token ) ) : #
            n_gram_sequence = token[ : t + 1 ] # 토큰 리스트에서 처음부터 t+1번째 단어까지를 슬라이싱를 한다 ( 시퀀스 )
            print( n_gram_sequence )
            input_sequences.append( n_gram_sequence ) # 각 시퀀스 저장
    return input_sequences , total_words # 리스트 반환 # 모든 시퀀스가 저장된 리스트 와 단어수 반환 # 2개 값이 저장된 튜플 1개 반환
    # 모든 프로그래밍 은 연산식 또는 함수는 항상 반환/결과 값은 1개이다.
input_sequences , total_words = get_tokens( review )
print( input_sequences[ 31 : 40 ] ) # 샘플 확인

# 단어 사전
print( f"감동 : { tokenizer.word_index['감동'] }" )
print( f"영화 : { tokenizer.word_index['영화'] }" )
print( f"코믹 : { tokenizer.word_index['코믹'] }" )

# 4. 패딩 : 모델이 시퀀스(문장)들을 학습할때 길이를 맞춤으로써 동일한 차원을 처리할수 있게 위해서 해야한다.
max_len = max( [ len(word) for word in input_sequences ]  ) # 모든 시퀀스 중에 가장 길이가 큰 수 찾기
print( max_len ) # 가장 긴 문장은 59개 단어를 가졌다.
# 패딩 함수를 이용한 패딩화 하기 # pad_sequences( 데이터리스트 , maxlen = 최대길이 , padding = 'pre앞post뒤' )
result = pad_sequences( input_sequences , maxlen= max_len , padding= 'pre' )
print( result )
import  numpy as np # 넘파이 객체
input_sequences = np.array( result ) # 패딩 결과를 다시 배열로 변환
print( input_sequences )

# 5. 독립변수 와 종속변수(정답) 구분하기 : 모델의 학습을 위해서
from tensorflow.keras.utils import to_categorical
# x는 독립변수 데이터 로 , 각 시퀀스의 마지막 단어를 제외함(왜?마지막단어는 예측하기위해)
# 즉] 모델은 시퀀스의 처음부터 마지막 단어 직전까지를 학습 시킨다.
x = input_sequences[ : , : -1 ] # 마지막 값은 제외함
# y는 종속변수 데이터 로 , 각 시퀀스의 마지막 단어를 원핫인코딩으로 변환한다.(왜? 위치 찾기위해서)
y = to_categorical( input_sequences[ : , -1 ] , num_classes=total_words) # 마지막 값만 이진 클래스 벡터로 변환
# * to_categorical( ) : 레이블 값을 원핫 인코딩 를 하여 반환 함수
a = to_categorical( [ 0 , 1 , 2 , 3 ] , num_classes= 4  )
# [ 0 , 1 , 2 , 3 ] 원핫인코딩 했을때 [ 1 0 0 0 ] [ 0 1 0 0 ] [ 0 0 1 0 ] [0 0 0 1]
print( a )

# 6. 모델 생성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , LSTM , Dense , Bidirectional , Dropout
embedding_dim = 256
model = Sequential()
model.add( Embedding( input_dim = total_words , output_dim= embedding_dim , input_length= max_len - 1  ) )
model.add( Bidirectional( LSTM( units = 256 ) ))
model.add( Dense( units = total_words , activation='softmax') )

model.compile( loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )
history = model.fit( x , y , epochs= 20  )




































