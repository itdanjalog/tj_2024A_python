
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import pandas as pd
import re

data = pd.read_csv('chatbot_conversations.csv')

print( data.head() )
# 입력과 출력 데이터 분리
inputs = list(data['Question'])
outputs = list(data['Response'])

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
model.add(Bidirectional(LSTM(64, return_sequences=True)))  # Bidirectional LSTM
model.add(Dropout(0.5))  # Dropout 추가
model.add(LSTM(32))  # 추가 LSTM 레이어
model.add(Dense(len(outputs), activation='softmax'))

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(input_sequences, output_sequences, epochs=200)

# 챗봇 응답 함수
def respond(user_input):
    processed_input = preprocess(user_input)
    sequence = tokenizer.texts_to_sequences([processed_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)
    response_index = np.argmax(prediction)
    return outputs[response_index]

# 챗봇과 대화하기
while True:
    user_input = input("사용자: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = respond(user_input)
    print("챗봇: ", response )