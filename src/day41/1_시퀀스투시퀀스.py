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
                # 예측결과에서 가장 확률이 높은 인덱스 찾기 : tf.argmax( decoder_output , axis=1 ) # tf.argmax() 가장 높은 값의 인덱스 반환함수
                decoder_output = tf.cast( tf.argmax( decoder_output , axis=1 ) , dtype=tf.int32 )
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

























