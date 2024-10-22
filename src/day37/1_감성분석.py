# day37 --> 1_감성분석 # python3.8
# 네이버 영화 리뷰 데이터
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 1. 데이터 다운로드
train_file = tf.keras.utils.get_file(
    'ratings_train.txt', # 다운로드된 파일의 이름 지정
    origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', # 파일을 다운로드할 URL 지정
    extract=True # 만약에 압축파일이면 자동으로 압축풀기 지정
)
# 2. 판다스 이용하여 해당 파일객체로 부터 파일 읽어오기 # \t 구분자
train = pd.read_csv( train_file , sep = '\t')
# 3. 읽어온 파일의 크기
print( train.shape ) # (150000, 3)
print( train.head() ) # id : 게시물번호 , # document : 리뷰명 , # label : 0부정1긍정
# 4. 레이블 별 개수  # pd객체['필드명'].value_counts() : 지정한 필드의 데이터별 개수
cnt = train['label'].value_counts()
print( cnt )
# 5. 레이블 별 비율 시각화
sns.countplot( x='label' , data=train )
plt.show()
# 6. 결측치( 데이터없는 / 빈 값 ) 확인 # pd객체.isnull()
print( train.isnull().sum() ) # 결측치 개수 확인
print( train[ train['document'].isnull() ] ) # 결측치가 있는 'document' 의 행 확인 # 추후에 결측치 행 삭제
# 7. 긍정/부정 텍스트 길이 시각화
fig , (ax1 , ax2 ) = plt.subplots( 1 , 2 , figsize = (10,5) )
data_len = train[train['label'] == 1 ]['document'].str.len() # pd객체내 'label' 필드값이 1 인 'document' 필드의 텍스트 길이 구하기
print( data_len )
ax1.hist( data_len ) # 히스토그램 차트
data_len = train[train['label'] == 0 ]['document'].str.len() # pd객체내 'label' 필드값이 0 인 'document' 필드의 텍스트 길이 구하기
print( data_len )
ax2.hist( data_len )
plt.show()
# 8. 형태소 분석기 객체 불러오기 # 형태소란 : 의미를 가지는 가장 단위 단위 # 즉] 더이상 쪼갤수 없는 최소의 의미 단위
# 오늘날씨어때 -> 오늘 , 날씨 vs 오늘날 , 씨 - 이처럼 띄어쓰기가 안 돼 있을때는 형태소 분석이 어렵다.
import konlpy # konlpy 패키지 설치
from konlpy.tag import Kkma , Komoran , Okt # Mecab 제외
kkma = Kkma() # Kkma 객체 생성
komoran = Komoran() # Komoran 객체 생성
okt = Okt() # Okt 객체 생성
# Mecab 생략

# 9. 형태소별 샘플
def sample_ko_pos( text ) :
    print('--------[text]--------')
    print(f'kkma : { kkma.pos(text) } ')
    print(f'komoran : {komoran.pos(text)} ')
    print(f'okt : {okt.pos(text)} ')
text = "영실아안녕오늘날씨어때?" # 띄어쓰기 없을때
# sample_ko_pos( text )
# kkma : [('영', 'MAG'), ('싣', 'VV'), ('아', 'ECD'), ('안녕', 'NNG'), ('오늘날', 'NNG'), ('씨', 'VV'), ('어', 'ECD'), ('때', 'NNG'), ('?', 'SF')]
# komoran : [('영', 'NNP'), ('실', 'NNP'), ('아', 'NNP'), ('안녕', 'NNP'), ('오늘날', 'NNP'), ('씨', 'NNB'), ('어떻', 'VA'), ('어', 'EF'), ('?', 'SF')]
# okt : [('영', 'Modifier'), ('실아', 'Noun'), ('안녕', 'Noun'), ('오늘날', 'Noun'), ('씨', 'Suffix'), ('어때', 'Adjective'), ('?', 'Punctuation')]
text2 = "영실아안뇽오늘날어때?" # 맞춤법이 틀렸을때
#sample_ko_pos( text2 )
# kkma : [('영', 'MAG'), ('싣', 'VV'), ('아', 'ECD'), ('안녕', 'NNG'), ('오늘날', 'NNG'), ('씨', 'VV'), ('어', 'ECD'), ('때', 'NNG'), ('?', 'SF')]
# komoran : [('영', 'NNP'), ('실', 'NNP'), ('아', 'NNP'), ('안녕', 'NNP'), ('오늘날', 'NNP'), ('씨', 'NNB'), ('어떻', 'VA'), ('어', 'EF'), ('?', 'SF')]
# okt : [('영', 'Modifier'), ('실아', 'Noun'), ('안녕', 'Noun'), ('오늘날', 'Noun'), ('씨', 'Suffix'), ('어때', 'Adjective'), ('?', 'Punctuation')]
text3 = "정말 재미있고 매력적인 영화에요 추천합니다."
sample_ko_pos( text3 )
# kkma : [('정말', 'MAG'), ('재미있', 'VA'), ('고', 'ECE'), ('매력적', 'NNG'), ('이', 'VCP'), ('ㄴ', 'ETD'), ('영화', 'NNG'), ('에', 'JKM'), ('요', 'JX'), ('추천', 'NNG'), ('하', 'XSV'), ('ㅂ니다', 'EFN'), ('.', 'SF')]
# komoran : [('정말', 'MAG'), ('재미있', 'VA'), ('고', 'EC'), ('매력', 'NNG'), ('적', 'XSN'), ('이', 'VCP'), ('ㄴ', 'ETM'), ('영화', 'NNG'), ('에', 'JKB'), ('요', 'JX'), ('추천', 'NNG'), ('하', 'XSV'), ('ㅂ니다', 'EF'), ('.', 'SF')]
# okt : [('정말', 'Noun'), ('재미있고', 'Adjective'), ('매력', 'Noun'), ('적', 'Suffix'), ('인', 'Josa'), ('영화', 'Noun'), ('에요', 'Josa'), ('추천', 'Noun'), ('합니다', 'Verb'), ('.', 'Punctuation')]

# 10. 데이터 전처리 #  mecab.morphs(text) -----> okt.morphs(text)
    # 1 한글과 영문을 제외한 모두 삭제/치환
train['document'] = train['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","" , regex=True ) # pd판다스버전 2.x이상부터 , regex=True 속성
print( train['document'].head() )
    # 2. 결측치/빈 데이터 제거 # .dropna( ) : 결측치 제거 함수
train = train.dropna( )
print( train.shape )
    # 3. 불용어 제거
def word_tokenization( text ) :
    # 불용어 목록 : 관사 , 전치사 , 조사 , 접속사 등  의미가 없는 단어를 제거
    stop_words = ['는' , '을' , '를' , '이' ,'가','의','던' ,'고' ,'하' ,'다' ,'은' ,'에' ,'들' ,'지' ,'게' ,'도' ]
    # 방법1
    '''
    list = []
    for word in okt.morphs( text ) : 
        if word not in stop_words : 
            list.append( word )
    return list 
    '''
    # 방법2 : 컴프리헨션 # Open Korean Text ( 한국어 형태소 분석기 객체 )
    # okt.morphs() : 분석결과를 리스트로 반환 vs okt.pos() : 분석결과를 튜플로반환
    return [  word for word in okt.morphs( text )  if word not in stop_words ] # 리스트 컴프리헨션
    # 실습 : 문장이 15,000 개라서 시간이 걸린다.
data = train['document'].apply(( lambda x : word_tokenization(x) ) ) # document 열에 데이터 하나씩 불용어제거 함수에 대입한다.
# data = train['document'] # 테스트용
print( data.head() )

# 11. 훈련용 과 테스트용 분할 # tip: 1. fit( split= ) 2. train_test_split( )
training_size = 120000
# train 분할
train_sentences = data[:training_size]
valid_sentences = data[training_size:]
# label 분할
train_labels = train['label'][:training_size]
valid_labels = train['label'][training_size:]

# 12. 단어 사전 만들기 # .fit_on_texts( ) :문자와 숫자(인덱스)를 매칭한다 # 문자를 숫자로 변환한다.
from tensorflow.keras.preprocessing.text import Tokenizer
totkenizer = Tokenizer() # 토크나이저 객체  생성
totkenizer.fit_on_texts( data ) # 토큰(단어) 사전 만들기 # 빈도수 기준
# print( f'단어 사전(단어 와 인덱스(숫자) 매칭 ) : { totkenizer.word_index }')
print( f'총 단어 개수 : { len( totkenizer.word_index )}')
    # 5회 이상 반복되는 단어 찾기
def get_vocab_size( threshold ) :
    cnt = 0
    for x in totkenizer.word_counts.values() :
        if x >= threshold :
            cnt += 1
    return cnt
vocab_size = get_vocab_size( 5 ) # 5회 이상 단어 찾기
print( vocab_size ) # 출현 5회 이상의 단어 들의  수
# 13 . <OOV> : 사전에 없는 단어 # 사전에 없는 단어는 <OOV> 표현한다.
oov_tok = "<OOV>"
vocab_size = 15000
totkenizer = Tokenizer( oov_token = oov_tok , num_words= vocab_size +1 )
totkenizer.fit_on_texts( data )
print( f'총 단어 개수 : { len( totkenizer.word_index )}')
# 14. 숫자 벡터로 변환
print( train_sentences[ : 2 ] )
train_sequences = totkenizer.texts_to_sequences( train_sentences )
valid_sequences = totkenizer.texts_to_sequences( valid_sentences )
print( train_sequences[ : 2 ] )
# 15. 문장 중에서 최대 길이 구하기  # 모든 문장의 길이를 맞추기 .
# 모든 문장들이 길이가 일치하면 모델 성능 도움 # 최대길이의 문자으로 일치화
max_length = max( len(x)  for x in train_sequences )
print( f'문장 최대 길이 : { max_length } ')
# 16. 문장 길이를 동일하게 맞춘다 # 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
trunc_type = 'post' # 길이를 초과하는 자르는 속성 # post 뒤에 자르기
padding_type = 'post' # 길이를 미달하는 경우 0으로 채우는 속성 # post 뒤에 채운다.
train_padded = pad_sequences( train_sequences , truncating = 'post' , padding= padding_type , maxlen= max_length )
valid_padded = pad_sequences( valid_sequences , truncating = trunc_type , padding= padding_type , maxlen= max_length )
import numpy as np
train_labels = np.asarray( train_labels ) # 배열로 변환
valid_labels = np.asarray( valid_labels ) # 배열로 변환
print( f'샘플 : { train_padded[ : 1 ] }')

# 17. 모델
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , LSTM , Embedding , Bidirectional

model = Sequential()
model.add( Embedding( vocab_size , 32 ) )
model.add( Bidirectional( LSTM( 32 , return_sequences=False )  ) ) # Bidirectional양방향 이면 유닛(뉴런) *2
model.add( Dense( 32 , activation='relu') )
model.add( Dense( 1 , activation='sigmoid') ) # 이진분류 에서 자주 사용하는 활성화함수 # 출력레이어

model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
print( model.summary() )

# 18. 학습
# 오류 : ValueError: `logits` and `labels` must have the same shape, received ((64, 40, 1) vs (64,)),
history = model.fit( train_padded , train_labels , # 훈련용
                     validation_data=(valid_padded,valid_labels), # 학습중 사용할 테스트용
                     batch_size=64 , # 모델이 한번에 처리할 데이터 수
                     epochs=10 , # 에포크
                     verbose=2 # 학습시 콘솔에 요약 정도 속성 , # 0:출력없음 1:진행률바 2:결과요약만
                     )
print( history ) # 최종 정확도 , 손실함수 확인
# 19. 아래 새로운 리뷰 텍스트의 감정 분석 하기 # 예측하기
new_reviews = [ '영화 정말 재미있다' , '정말 지루하다' , '그냥 보통 이었어요.' , '생각보다 재미가 없다' ]
# 새로운 리뷰도 전처리
new_sequences = totkenizer.texts_to_sequences( data ) # .texts_to_sequences() 토크나이저 있는 사전 기반으로 문자를 숫자(벡터) 변환
new_padded_sequences = pad_sequences( new_sequences , maxlen=max_length ) # .pad_sequences() 문자열길이 맞춤 함수
# 모델 이용한 감성 예측
result = model.predict( new_padded_sequences )
# 예측 결과
for index , review in enumerate( new_reviews )  : # for 인덱스,반복변수 in enumerate(반복할객체) :
    print( f'리뷰 : {review}  , 확률 : { result[index] }' ) # 0 ~ 1 사이의 비율 # 0.5초과 긍정 # 0.5미만 부정











































