
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import pandas as pd
import re

# 샘플 대화 데이터를 리스트 안의 딕셔너리 형식으로 만듭니다.
data = [
    {"user": "안녕하세요", "bot": "안녕하세요! 무엇을 도와드릴까요?"},
    {"user": "오늘 날씨 어때요?", "bot": "오늘은 맑고 화창한 날씨입니다."},
    {"user": "지금 몇 시에요?", "bot": "현재 시간은 오후 3시입니다."},
    {"user": "좋은 책 추천해 주세요", "bot": "최근에 인기가 많은 책은 '파이썬 데이터 분석'입니다."},
    {"user": "고마워요", "bot": "천만에요! 더 필요한 것이 있으면 말씀해주세요."}
]

# 리스트의 딕셔너리 데이터를 DataFrame으로 변환
data = pd.DataFrame(data)


print( data.head() )
# 입력과 출력 데이터 분리
inputs = list(data['user'])
outputs = list(data['bot'])

# 형태소 분석기 초기화
okt = Okt()

# 불용어 목록
# stopwords = ['은', '는', '이', '가', '을', '를', '에', '의', '도', '와', '한', '하다', '고', '들']



# 형태소 분석 및 불용어 제거를 포함한 전처리 함수
def preprocess(text):
    # 한글 이외의 문자 제거
    cleaned_text = re.sub(r'[^가-힣\s]', '', text)
    print( cleaned_text )

    # 형태소 분석 (품사 태그까지 추출)
    morphs_with_pos = okt.pos(cleaned_text, stem=True)  # 어간 추출(stem=True)로 어미를 제거하고 일반화
    print( morphs_with_pos )

    # 명사와 동사, 형용사만 남기기
    morphs_filtered = [word for word, pos in morphs_with_pos if pos in ['Noun', 'Verb', 'Adjective']]
    print( morphs_filtered )

    # 불용어 제거
    # morphs_cleaned = [word for word in morphs_filtered if word not in stopwords]

    # 중복 공백 제거 후 결과 반환

    return ' '.join(morphs_filtered).strip()

# 입력 데이터 전처리
processed_inputs = [preprocess(sentence) for sentence in inputs]

# 전처리된 데이터 확인
print(processed_inputs[:5])


# 토크나이저 정의
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_inputs)

# 입력 시퀀스 변환
input_sequences = tokenizer.texts_to_sequences(processed_inputs)
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length)

# 출력 데이터 인코딩
output_sequences = np.array(range(len(outputs)))

# 모델 정의
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_sequence_length))
model.add( Bidirectional( LSTM( 256 ) ) ) ,  #  256 , 128 , 64 , 32
model.add(Dense(len(outputs), activation='softmax'))

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(input_sequences, output_sequences, epochs=10)

def response( text ) :
    text = preprocess( text )# 1. 예측할 값도 전처리 한다.
    text = tokenizer.texts_to_sequences( [ text ] )  # 2. 예측할 값도 토큰 과 패딩  # 학습된 모델과 데이터 동일
    text = pad_sequences( text , maxlen= max_sequence_length )
    result = model.predict( text ) # 3. 예측
    max_index = np.argmax( result )  # 4. 결과 # 가장 높은 확률의 인덱스 찾기
    return outputs[max_index]  # 5.
# 확인
print( response('안녕하세요') ) # 질문이 '안녕하세요' , 학습된 질문 목록중에 가장 높은 예측비율이 높은 질문의 응답을 출력한다.
# 서비스 제공한다. # 플라스크
while True :
    text = input( '사용자 : ' ) # 챗봇에게 전달할 내용 입력받기
    result = response( text ) # 입력받은 내용을 함수에 넣어 응답을 예측를 한다.
    print( f'챗봇 : { result }') # 예측한 응답 출력한다.