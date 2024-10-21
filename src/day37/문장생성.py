import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import random

# 1. 데이터 로드
train_file = tf.keras.utils.get_file(
    'ratings_train.txt',
    origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',
    extract=True
)

# 2. 데이터 로드 및 전처리
data = pd.read_csv(train_file, sep='\t')
data = data.dropna(how='any')  # 결측값 제거

# 텍스트만 추출
texts = data['document'].values

# 샘플링: 상위 10,000개 리뷰만 사용
texts = texts[:10000]  # 또는 더 적은 수의 리뷰로 줄일 수 있습니다.

# 3. 토큰화
tokenizer = Tokenizer(num_words=10000)  # 최대 단어 수 10,000
tokenizer.fit_on_texts(texts)

# 총 단어 수
total_words = len(tokenizer.word_index) + 1

# 4. 텍스트 시퀀스 생성
input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 시퀀스 패딩
max_sequence_len = 20  # 최대 시퀀스 길이 20으로 제한
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# 입력(X)과 출력(y) 분리
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# 출력 y를 원-핫 인코딩
y = to_categorical(y, num_classes=total_words)

# 5. 모델 설계
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    LSTM(150),
    Dense(total_words, activation='softmax')
])

# 6. 모델 컴파일 및 학습
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=64, verbose=1)  # 배치 사이즈 64로 설정

# 7. 문장 생성 함수
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)
        output_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + output_word
    return seed_text

# 8. 문장 생성 예시
seed_text = "이 영화는"
next_words = 10  # 생성할 단어 수
generated_sentence = generate_text(seed_text, next_words, max_sequence_len)
print(generated_sentence)
