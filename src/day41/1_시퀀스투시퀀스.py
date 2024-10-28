# day39 --> 1_시퀀스투시퀀스.py # python3.8 gpu

import pandas as pd
# 1. 데이터수집
# - 질문과 답변이있는 말뭉치(대화내용)를 가져오기
    # pd.read_csv('csv로컬경로/웹경로')
    # Q(질문),A(답변),label(0:일상다반사,1:부정 2:긍정)
corpus = pd.read_csv('https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv')
# - 확인
print( corpus['Q'].head() ) # 질문 '열' 의 상단 5개 데이터 확인
print( corpus['A'].head() ) # 답변 '열' 의 상단 5개 데이터 확인
# - 확인 : 같은 인덱스로 Q와 A로 구성됨.
print( f"Q : {corpus['Q'][0]} ") # Q 12시 땡!
print( f"A : {corpus['A'][0]} ") # A 하루가 또 가네요.
# - 확인
print( corpus.shape ) # (11823, 3) # (행,열) # .shape: 데이터프레임 객체의 차원 확인 속성
# - 샘플링( 1,000 개 사용 )
texts = [] # 질문 리스트
pairs = [] # 답변 리스트
# for index , value in enumerate( 리스트/튜플 ) :
# for value in 리스트/튜플 :
print( zip(corpus['Q'] , corpus['A'] )  ) # zip( 리스트 , 리스트 )
print( type( zip(corpus['Q'] , corpus['A'] ) )  )
for Q , A in zip( corpus['Q'] , corpus['A'] ) :
    print( f'Q : {Q}   , A : {A}' )
for i , (text , pair ) in enumerate( zip(corpus['Q'] , corpus['A'] ) ) :
    texts.append( text )
    pairs.append( pair )
    if i >= 1000 : # 1000개 의 인덱스만 사용 # 샘플링 # RAM문제
        break
# 2. 데이터 전처리
import re # 파이썬 문자열 정규표현식 객체
# - 정규표현식
def clean_sentence( sentence ) :
    # 한글 , 숫자를 제외한 모든 문자는 제거
    # 1. re.sub(r'정규표현식', r'대체할문자' , 문자열 ) : 파이썬 내장용 문자열 정규표현식 함수
    # 2. pd['열이름'].str.replace("정규표현식","" , regex=True ) : 데이터프레임내 정규표현식 방법
    return re.sub( r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣\s]',r'',sentence)
# - 확인
print( clean_sentence('안녕하세요~:') ) # 안녕하세요
print( clean_sentence('텐서플로^@^%#@!')) # 텐서플로
# - 한글 형태소 분석기
from konlpy.tag import Okt
okt = Okt() # 한글분석기 객체
def process_morph( sentence ) :
    return ' '.join( okt.morphs( sentence ) ) # 형태소 분석 결과 목록 을 하나의 문자열 합치기
    # 형태소들 사이에 공백' ' 으로 구성한 문자열
print( '안녕하세요'.join(['유재석' , '강호동'] ) ) # '유재석안녕하세요강호동'
# - 전처리 실행후 질문전체 , 답변시작 , 답변끝 구분
def clean_and_morph( sentence , is_question=True ) : # 매개변수명=초기값 : 매개변수에 초기값 넣기
    # 한글 문자 ( 정규표현식 함수 실행 )
    sentence = clean_sentence( sentence )
    # 형태소 변환( 형태소 함수 실행 )
    sentence = process_morph( sentence )
    # 질문(Question) 인 경우 , Answer(답변) 인 경우를 구분하여 처리
    if is_question :
        return sentence
    else : # 프로그래밍 언어에서 함수는 무조건 리턴(결과) 1개 이다.
        return (f'<START> {sentence}' , f'{sentence} <END>') # ( 값1 , 값2 ) : 튜플 형식 # ( )생략 가능
# - 질문전체 , 답변시작 , 답변끝 리스트 만들기
def preprocess( texts , pairs ) :
    questions = [] # 인코더에 입력할 질문 전체 리스트
    answer_in = [] # 디코더에 입력할 답변의 시작 , <START> 토큰을 문장 처음에 추가 , # 데이터들을 구분한 단위 : 토큰
    answer_out = [] # 디코더에 출력할 답변의 끝 , <END> 토큰(단어)를 문장 끝에 추가
    # 질의에 대한 전처리
    for text in texts :
        question = clean_and_morph( text , is_question =True ) # is_question=True 질의
        questions.append( question ) # 질문을 질문 목록에 담는다.
    # 답변에 대한 전처리
    for pair in pairs :
        ( in_ , out_ ) = clean_and_morph( pair , is_question = False ) # , is_question= False 답변
        answer_in.append( in_ )
        answer_out.append( out_ )
    return questions , answer_in , answer_out # 질문전체리스트,답변시작,답변끝
(questions , answer_in , answer_out ) = preprocess( texts , pairs )
print( questions[ : 2] ,  answer_in[ : 2] , answer_out[ : 2] )
# ['12시 땡', '1 지망 학교 떨어졌어'] ['<START> 하루 가 또 가네요', '<START> 위로 해 드립니다'] ['하루 가 또 가네요 <END>', '위로 해 드립니다 <END>']

# 전체 문자을 하나의 리스트로 만들기 #
all_sentences = questions + answer_in + answer_out
import  numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings('ignore') # 경고 무시
# - 단어 사전 만들기
    # filters='' : 토큰화 할때 특정 기호를 제거(필터) # 필터링 하지 안겠다는 뜻
    # lower = False : 토큰화 할때 소문자로 변환하지 여부 # 기본값 true 이므로 모든 영문을 소문자로 변환
        # false 이므로 변환하지 않는다.
    # oov_token : 단어 사전에 없는 단어를 매칭할때 그 단어를 대체할 문자<OOV> 표현
tokenizer = Tokenizer( filters='' , lower=False , oov_token='<OOV>') # 변수명 = 클래스명(  )
tokenizer.fit_on_texts( all_sentences )
print( tokenizer.word_index ) # 단어 사전 확인
# {'<OOV>': 1, '<START>': 2, '<END>': 3, '이': 4, '거': 5, '을': 6, '가': 7, '나': 8, '예요': 9, '사람': 10, '요': 11, '도': 12, ~~ }
print( len( tokenizer.word_index ) ) # 단어의 총 개수 # 2302
# .texts_to_sequences() 등록된 단어사전에 따라 문장의 단어들을 벡터(숫자) 매칭하여 변환
question_sequence = tokenizer.texts_to_sequences( questions );      print( question_sequence[0] ) # [666]
answer_in_sequence = tokenizer.texts_to_sequences( answer_in );     print( answer_in_sequence[0] ) # [2, 317]
answer_out_sequence = tokenizer.texts_to_sequences( answer_out );   print( answer_out_sequence[0] ) # [317, 3]
# 패딩 (문장의 길이 맞추기 : 학습 데이터들의 차원 일치화 함으로 모델 성능 향상 )
MAX_LENGTH = 30 # 문장내 최대 단어 개수(길이) 는 임의로 30 # post : 빈칸을 뒤에 0으로 채우기
question_padded = pad_sequences( question_sequence , maxlen=MAX_LENGTH , padding='post' )
answer_in_padded = pad_sequences( answer_in_sequence , maxlen=MAX_LENGTH , padding='post' )
answer_out_padded = pad_sequences( answer_out_sequence , maxlen=MAX_LENGTH , padding='post' )
print( question_padded.shape , answer_in_padded.shape , answer_out_padded.shape ) # (1001, 30) (1001, 30) (1001, 30)

# 상속 : 하나의 클래스가 다른 클래스에게 속성/필드 과 함수/기능 물려두는 행위
    # 자바 : class 클래스A extends 클래스B{ }
        # this , super
    # 파이썬 : class 클래스A( 클래스B ) :
        # self , super
from tensorflow.keras.layers import Embedding , LSTM , Dense , Dropout
from tensorflow.keras.models import Model

# - 텐서플로의 Model 클래스로부터 상속받아 인코더 클래스 정의하기
class Encoder( tf.keras.Model ) :
    # 초기화함수 # 생성자 # 사용할 변수 , 레이어를 미리 불러와서 파라미터 값들을 미리 설정 한다.
    def __init__(self , units , vocab_size , embedding_dim , time_steps ):
                # units 매개변수1 : LSTM에서 사용할 유닛/노드/뉴런 수
                    # "안녕하세요, 오늘 날씨 어떄요?" 문장 가정이라고 했을때.
                # vocab_size 매개변수2 : 임베딩 레이어의 입력으로 들어가는 단어 크기
                    # "안녕하세요" , "오늘" , "날씨" ,"어때요" => 4
                # embedding_dim 매개변수3 : 임베딩 레이어의 각 단어를 크기의 벡터 차원
                    # 밀집행렬를 처리할때 한 단어를 표현을 차원수 # "안녕하세요" 몇차원으로 구성할지
                # time_steps 매개변수4 : 임베딩 레이어의 입력으로 들어가는 시퀀스의 길이
                    # 한번에 몇개의 단어를 모델이 학습하고 기억할지 단위 길이 # 2 => "안녕하세요" , "오늘"
        super( Encoder , self ).__init__() # 상속받은 슈퍼클래스의 초기화함수(생성자) 를 호출
        # 1. 임베딩 레이어
        self.embedding = Embedding( vocab_size , embedding_dim , input_length=time_steps )
        # 2. 드롭아웃 레이어 # 일반드롭아웃 # 0.2 : 20%를 무작위로 비활성
        self.dropout = Dropout( 0.2 )
        # 3. LSTM 레이어 #
        self.lstm = LSTM( units , return_state= True )
    # 실행 함수 # call
    def call(self, inputs ):
        x = self.embedding( inputs ) # 임베딩 레이어 에 따른 밀집행렬(벡터) 하기
        x = self.dropout( x ) # 드롭아웃 레이어 에 따른 무작위 노드를 제외 하기
        x , hidden_state , cell_state = self.lstm( x ) # LSTM 레이어 에 따른 학습
            # x : LSTM 알고리즘이 특정 단어로 부터의 특징(정보/패턴) 값
                # 문장 : '오늘 무엇을 먹을까?' ---> 현재 문장의 분석 결과를 알려주는 출력값
            # 은닉 상태 : LSTM 알고리즘이 현재 시점에서의 기록한 특징들(정보/패턴)들을 저장하는 메모리
                # L(LONG)S(SHORT)T(TERM)M : 앞전 문장을 잊지 않고 지속하는 문장을 기록하는 메모리
            # 셀 상태 : LSTM 알고즘이 전체 단어들 에서 중요한 특징(정보/패턴)들을 저장하는 메모리
                # 앞전 전체 분석된 문장들 중에서 중요한 단어들을 기억하는 메모리
            # (특징/패턴) 분석
                # CNN : 이미지 분석 , # 곡선 , 색감 , 사이즈 , 비율 , 질감(텍스처) 등등 # 0~255 # 컴퓨터는 이미지를 RGB
                # RNN : 텍스트 분석 , # 빈도 , 감정 , 형태소(동사,형용사 등등) , 단어의 의미 # 벡터 # 컴퓨터는 텍스트 대신 벡터
        # Dense 레이어가 없는 이유는 현재 클래스(인코더) 의 목적은 입력과정 하기 위해서 --> 디코더 전달할 예정
        return [ hidden_state , cell_state ]

# - 텐서플로의 Model 클래스로부터 상속받아 디코더 클래스 정의하기
class Decoder( tf.keras.Model ) :
    # 1.
    def __init__(self , units , vocab_size , embedding_dim , time_steps ):
        super( Decoder , self ).__init__()
        self.embedding = Embedding( vocab_size , embedding_dim , input_length=time_steps)
        self.dropout = Dropout( 0.2 )
        self.lstm = LSTM( units , return_state=True , return_sequences=True )
            # return_state=True : 생략가능(기본값) , 은닉상태와셀상태 반환 설정
            # return_sequences=True : 모든 시점의 출력을 반환한다.
        self.dense = Dense( vocab_size , activation='softmax')# 최종 출력 레이어
    # 2.
    def call(self , inputs , initial_state):
        x = self.embedding( inputs )
        x = self.dropout( x )
        x , hidden_state , cell_state = self.lstm( x , initial_state= initial_state ) # LSTM 레이어 에 따른 학습
            # initial_state : 초기화상태 속성 # 인코더와 결합 이후에 인코더에 생성한 은닉상태 와 셀 상태를 대입한다.
        x = self.dense( x ) # 출력 레이어 # 출력 : 학습된 모델에서의 최종 출력된 값 : X
        return ( x , hidden_state , cell_state ) # ( 최종확률값 , 은닉상태 , 셀상태 )


# day39 ---> day41 복사 붙여넣기 - 329P.
# =============================================================================================
class Seq2Seq( tf.keras.Model ) : # 클래스 정의
    # 1. 초기화 함수 # 객체 생성자 함수
    def __init__( self , units , vocab_size , embedding_dim , time_steps , start_token , end_token ):    # self : 자신 객체 뜻
        super( Seq2Seq , self ).__init__()
        self.start_token = start_token # 객체의 속성를 정의후 매개변수 대입 # 시작 토큰 # 모델이 문장을 생성할때 시작을 식별하기 위해 사용
        self.end_token = end_token # 끝나는 토큰 # 모델이 문장을 생성할때 끝마침을 식별하기 위해 사용
        self.time_steps = time_steps
        self.encoder = Encoder( units , vocab_size , embedding_dim , time_steps ) # 인코더 객체 생성 # 각 매개변수 대입
        self.decoder = Decoder( units , vocab_size , embedding_dim , time_steps ) # 디코더 객체 생성 # 각 매개변수 대입
    # 2. 실행 함수 # 객체 호출 함수
    def call( self , inputs , training = True  ):
        # inputs : 모델객체 안으로 들어오는 입력 데이터
        # training = True : true 훈련중 일때 , false 훈련중이 아닐때 # 매개변수 = 초기값 # 훈련 중을 기본값으로 사용한다.
        if training : # 훈련중이면 # fit()
            encoder_inputs , decoder_inputs = inputs # 현재 모델이 주어진 인코더와 디코더의 입력 # fit() 메소드 호출시 들어오는 데이터
            context_vector = self.encoder( encoder_inputs ) # encoder 객체의 call 함수 호출 하고 결과 받기
            decoder_outputs , _ , _ = self.decoder( inputs=decoder_inputs , initial_state = context_vector ) #
            # decoder 객체의 call 함수 호출 하고 결과 받기 # initial_state : 인코더 결과 값
                # _(언더바) : 변수 생략  # for _ in 리스트  # ( 값 , _ , _ ) = 함수( )
            return decoder_outputs # 디코더의 출력을 반환 # 최종 예측한 확률값

        else : # 훈련이 아닐떄 # 예측모드 # 추론모드 # 문장을 생성 하기 위한 생성할 문장을 예측 하는 코드
            context_vector = self.encoder( inputs )
            target_seq = tf.constant( [[ self.start_token ]] , dtype=tf.float32 )
            # 시작 토큰을 이용한 2차원 텐서를 생성한다. # tf.constant() : 차원 만들기 함수
            results = tf.TensorArray( tf.int32 , self.time_steps )
            # 디코더의 출력 결과를 저장하기 위한 배열 생성한다. # tf.TensorArray() 텐서 배열 만들기 함수

            # 디코더가 다음 단어( 문장 만들기 ) 를 예측하는 과정 반복
            for i in tf.range( self.time_steps ) :
                # ( 최종확률값 , 은닉상태 , 셀상태 ) = 디코더객체
                decoder_output , decoder_hidden , decoder_cell = self.decoder( target_seq , initial_state = context_vector )
                # 예측결과에서 가장 확률이 높은 인덱스 찾기 : tf.argmax( decoder_output , axis=-1 ) # tf.argmax() 가장 높은 값의 인덱스 반환함수 # tf.cast() 자료형 변환 함수
                decoder_output = tf.cast( tf.argmax( decoder_output , axis=-1 ) , dtype=tf.int32 )
                # tf.reshape() 차원을 변경 함수
                decoder_output = tf.reshape( decoder_output , shape=( 1 , 1 ) )
                # (TensorArray).write() : i번째 인덱스의 예측한 단어를 텐서배열에 저장
                results = results.write( i , decoder_output )
                # 예측한 단어가 종료 토큰( <END> 포함 )과 일치하면 반복문 종료
                if decoder_output == self.end_token :
                    break
                # 종료 토큰이 아니면
                target_seq = decoder_output # 현재 예측된 단어를 다음 반복에 입력으로 사용한다.
                context_vector = [ decoder_hidden , decoder_cell ] # 디코더의 상태를 업데이터 하여 다음 반복에 사용한다.
            # 반복문이 종료되면 # 모든 예측 결과를 스택 으로 반환한다. 이 결과에 모든 시퀀스를 예측한 결과를 포함
            return tf.reshape( results.stack() , shape= ( 1 , self.time_steps ) )
# 시퀀스란 : 문장을 정해진 순서대로 나열된 단어를 의미 # 알고리즘,자료구조,딥러닝 등등 에서 사용되는 용어
# 예] 반가워 사랑해 좋아 => ["반가워" , "사랑해" , "좋아" ]


VOCAB_SIZE = len( tokenizer.word_index ) +1 # tokenizer.word_index 단어사전 # 단어사전의 단어 개수 # +1 : <OOV> 추가 했으므로

# - 디코더의 결과를 원핫 인코딩 벡터로 변환
# 컴퓨터가 이해하는 언어인 벡터로 변환하는 방법
# 임베딩(밀집행렬) : 주로 챗봇의 질문에서 사용된다.( 학습 데이터 ) # 임베딩은 단어 간의 유사성 파악 유리
# vs
# 원핫인코딩 : 주로 챗봇의 답변에서 사용된다. ( 결과 데이터 ) # 유사성 파악 아닌 단순 분류 에서 유리
# 사과는 너무 맛있다.
# 사과 = [ 1  0  0 ]
# 너무 = [ 0  1  0 ]
# 맛있다 = [ 0  0  1 ]
def convert_to_one_hot( padded ) :
    # 1. 응답 개수 만큼의 차원수을 0 으로 채우기
    one_hot_vector = np.zeros( (len( answer_out_padded ) , MAX_LENGTH , VOCAB_SIZE )) # :np.zeros()
    # ( 데이터1, 데이터2 , 데이터3 ) : 3차원 배열을 초기화
    # len( answer_out_padded ) : 총 응답의 개수 # (1001, 30)
    # MAX_LENGTH : 문장내 최대 길이
    # VOCAB_SIZE : 단어 사전의 단어수
    # ( 응답단어의 총개수 , 최대길이 , 단어사전의 단어수 )
    # 2.지정한 인덱스의 1 채움 으로써 원 핫 인코딩을 완성한다.
        # 1. 행
    for i , sequence in enumerate( answer_out_padded ) : # for index , value in enumerate( 리스트 ) :
        # 2. 열
        # i : 현재 시퀀스의 인덱스 # sequence : 현재 시퀀스의 단어
        for j , index in enumerate( sequence ) :
            # 3. 높이
            # j : 현재 단어의 인덱스 # 현재 단어의 인덱스 번호
            one_hot_vector[ i , j , index ] = 1 # 지정한 인덱스의 1 채움
    return one_hot_vector
# - .zeros( 차원수 ) : 지정한 차원수 만큼 0 으로 채워진다.
# 1. (np).zeros( 5 ) : [ 0 0 0 0 0 ]
# 2. (np).zeros( 3 , 4 ) : [ [ 0 0 0 0 ] [ 0 0 0 0 ] [ 0 0 0 0 ] ]
# 3. (np).zeros( 2 , 3 , 4 ) : [ [ [ 0 0 0 0 ] [ 0 0 0 0 ] [ 0 0 0 0 ] ]  [ [ 0 0 0 0 ] [ 0 0 0 0 ] [ 0 0 0 0 ] ]  ]
answer_in_one_hot = convert_to_one_hot( answer_in_padded )
answer_out_one_hot = convert_to_one_hot( answer_out_padded )

# 모델(디코더) 이 예측한 단어목록(indexs:예측한단어의인덱스)를 이용한 새로운 문장 만들기 함수
def convert_index_to_text( indexs , end_token ) :
    sentence = '' # 생성된 문장을 저장할 변수를 선언 # 처음에는 빈 문자열
    for index in indexs :  # indexs 배열의 각 인덱스를 반복 # 해당 배열에는 예측된 단어가 위치한 배열
        if index == end_token : # 만약에 현재 인덱스가 end_token(마지막문장) 이면 문장 생성 종료한다.
            break
        # 예측한 인덱스가 0보다 크고 (토큰나이저) 단어사전내 지정한 인덱스의 단어가 None 이 아니면
        if index > 0 and tokenizer.index_word[index] is not None :
            sentence += tokenizer.index_word[index] # 찾았으면 찾은 단어를 생성한문장 변수에 += 누적으로 더한다.
        else : # 단어사전에 없는 인덱스이면 빈 문자열 추가
            sentence +=''
        # 빈칸 추가 # 다음 반복으로(다음 단어 생성) 이동 하기 전에 띄어쓰기 추가
        sentence += ' ' # 공백 추가
    # 전체 반복문이 종료
    return sentence # 생성된 문장(변수) 반환

###  .ckpt --> .weights.h5 : 확장자 변경
# 모델 객체 생성 하기전에 파리미터 값 정의
BUFFER_SIZE = 1000 # 버퍼 : 모델이 훈련 중에 저장할 (무작위)샘플 최대수
# 버퍼가 클수록 다양하게 잘 섞여서 학습에 성능 향상 하는데, 메모리 소모가 크다. # 조절
BATCH_SIZE = 16 # 배치 : 모델이 훈련 중에 훈련1번에 있어서 사용되는 샘플 수
# 배치가 클수록 안정적이지만, 메모리 소모가 크다 # 8 16 32 64 단위로 주로 사용된다. # 조절
EMBEDDING_DIM = 100 # 임베딩 차원 : 단어를 벡터로 인코딩 과정, 인코딩 과정에 있어서 한 단어가 사용할 차원수
# 벡터로 표현할 차원수가 크면 표현성능이 좋아지지만 # 메모리 소모 와 계산 비용이 증가한다. # 단어들간의 의미 관계 파악할수 있다.
TIME_STEPS = MAX_LENGTH # 문장내 단어의 최대 개수 # 30(임의)
START_TOKEN = tokenizer.word_index['<START>'] # 문장의 시작을 알리는 토큰(단어) 인덱스 # 단어 생성시(예측) 시작 위치
END_TOKEN = tokenizer.word_index['<END>'] # 문장의 끝을 알리는 토큰(단어) 인덱스 # 단어 생성시(예측) 해당 토큰을 만나면 문장생성 종료
UNITS = 128 # 유닛 수 : RNN(유닛) CNN(노드) ==> 뉴런 수 # 각 모델이 학습하는 레이어에 사용될 뉴런 수
# 많은 유닛 수를 사용하면 더 복잡한 학습이 가능 하지만 , 과대적합에 빠질수 있다 , 주로 32 , 64 , 128 , 256 단위로 사용한다.
VOCAB_SIZE = len( tokenizer.word_index) + 1 # (토큰나이저)단어사전내 단어 수 # +1 : <OOV>
NUM_EPOCHS = 20 # 훈련 반복 횟수

DATA_LENGTH = len( questions ) # 질문의 총 개수
SAMPLE_SIZE = 3 # 샘플 개수

# 모델의 가중치를 저장하고 추후에 가중치를 재 호출하여 다른 모델 또는 곳 에서 재 사용
checkpoint_path = 'model/no-attention.weights.h5' # 경로/파일명.weights.h5 # 현재 폴더내 'model' 생성
from tensorflow.keras.callbacks import ModelCheckpoint # 체크포인트 클래스 모듈 가져오기
checkpoint = ModelCheckpoint( filepath= checkpoint_path , # 1. 모델 가중치를 저장할 파일 경로 지정
                              save_weights_only = True ,  # 2. 모델의 가중치만 저장 # True 모델의 구조는 저장되지 않는다 # 구조 저장은 False
                              save_best_only = True ,     # 3. 훈련중 모니터( fit : var_loss ) 값이 개선될때 만 가중치를 저장 # 성능이 향상될때 체크포인트 업데이트
                              monitor='loss' ,            # 4. 어떤 값을 모니터링 할지 지정 # loss(손실함수)
                              verbose= 1 )                # 5. 과정 로그 수준 # 생략 가능
# 시퀀스 모델 객체 생성
seq2seq = Seq2Seq( UNITS , VOCAB_SIZE , EMBEDDING_DIM , TIME_STEPS , START_TOKEN , END_TOKEN )
# 모델 컴파일
seq2seq.compile( optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])
# 모델 학습후 예측 함수
def make_prediction( model , question_inputs ) : # model : 학습한 모델 , question_inputs : 예측할 새로운 질문
    results = model( inputs = question_inputs , training=False) # 예측이므로 훈련이 아니다. # Seq2Seq클래스내 call함수내 else 코드들이 실행된다.
    # 변환된 인덱스를 문장으로 변환
    results = np.asarray( results ) .reshape( -1 )
    # 예측 결과를 np(넘파일) 배열로 변환 하고 차원을 1차원(-1) 배열로 변경한다.
    # 나중에 문장 조회시 평탄화(1차원변경)하고 convert_index_to_text() 에게 전달할 예정
    return  results

# 훈련 과정
for epoch in range( NUM_EPOCHS ) : # 총 20회 반복
    print( f' processing epoch : { epoch * 10 + 1 } ') # 현재 에포크
    seq2seq.fit( [question_padded , answer_in_padded ] , answer_out_one_hot ,
                 epochs = 10 , batch_size= BATCH_SIZE , callbacks=[checkpoint] )
    # fit() 모델 훈련 함수
    # 1. [question_padded , answer_in_padded ] : 입력 데이터
    # 2. answer_out_one_hot : 결과 데이터
    # 3. callbacks : 훈련중 체크포인트 지정한다. # 가중치만 저장

    # 훈련후 샘플수 만큼 난수의 질문을 이용하여 성능 예측하기
    samples = np.random.randint( DATA_LENGTH , size = SAMPLE_SIZE ) # 전체 질문에서 3개의 질문을 난수로 추출
        # np.random.randint(  전체수 , size = 추출할개수 ) : 0부터 전체수 까지 추출할 개수만큼 정수배열로 반환 함수
    # 예측 성능 테스트
    for index in samples : # 임의의 3개의 질문이 있는 리스트
        question_inputs = question_padded[index] # 선정된 질문의 인코딩(패딩) 된 단어 가져오기
        # 예측 # np.expand_dims( 배열 , 추가할차원인덱스 ) : 새로운 차원 추가 # 0 : 첫번째 자리에 차원 추가
            # ( 1 , 단어의패딩값 ) : 2차원으로 배열 만든다. # 모델의 예측 매개변수가 2차원이라서 차원 맞추기 ( 응답차원=0 , 입력차원  )
        results = make_prediction( seq2seq , np.expand_dims( question_inputs , 0 ) )
        # 예측한 벡터들을 문장으로 변환
        results = convert_index_to_text( results , END_TOKEN )
        # 확인
        print( f'Q : {questions[index]}')
        print( f'A : { results }')
        print( )



























