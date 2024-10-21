# 기본 라이브러리
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sympy import primenu

# 네이버 영화리뷰 데이터셋 불러오기
file = tf.keras.utils.get_file(
    'ratings_train.txt',
    origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',
    extract=True)

df = pd.read_csv(file, sep='\t')

# 데이터 임의샘플 확인
print( df[1000:1007] )

# 형태소 분석기
# !pip install konlpy
from konlpy.tag import Okt
okt = Okt()

# 데이터 전처리
def word_tokenization(text):
  return [word for word in okt.morphs(text)]

def preprocessing(df):
  df = df.dropna()
  df = df[1000:2000]  # 샘플 데이터 1000개, 학습시간을 줄이고자 함
  df['document'] = df['document'].str.replace("[^A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ ]","")
  data =  df['document'].apply((lambda x: word_tokenization(x)))
  return data

# 텍스트 데이터 1000개 전처리 후 불러오기
review = preprocessing(df)
len(review)

# 형태소 분리된 데이터 확인
print(review[:10])

# 토큰화 및 패딩
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()

def get_tokens(review):
  tokenizer.fit_on_texts(review)
  total_words = len(tokenizer.word_index)+1
  tokenized_sentences = tokenizer.texts_to_sequences(review)

  input_sequences = []
  for token in tokenized_sentences:
    for t in range(1, len(token)):
        n_gram_sequence = token[:t+1]
        input_sequences.append(n_gram_sequence)

  return input_sequences, total_words

input_sequences, total_words = get_tokens(review)
input_sequences[31:40] # n_gram으로 리스트된 데이터샘플 확인

# 단어 사전
print("감동 ==>> ",tokenizer.word_index['감동'])
print("영화 ==>> ",tokenizer.word_index['영화'])
print("코믹 ==>> ",tokenizer.word_index['코믹'])


# 문장의 길이 동일하게 맞추기
max_len = max([len(word) for word in input_sequences])
print("max_len:", max_len)
input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_len,
                                         padding='pre'))

# 입력텍스트와 타겟
from tensorflow.keras.utils import to_categorical
X = input_sequences[:,:-1]  # 마지막 값은 제외함
y = to_categorical(input_sequences[:,-1],
                   num_classes=total_words) # 마지막 값만 이진 클래스 벡터로 변환

# y를 설명하기 위한 예시
a = to_categorical([0, 1, 2, 3], num_classes=4)
a

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

embedding_dim = 256

model = Sequential([
    Embedding(input_dim=total_words,
              output_dim=embedding_dim,
              input_length=max_len-1),
    Bidirectional(LSTM(units=256)),
    Dense(units=total_words, activation='softmax'),
])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X, y, epochs=5, verbose=1) # 20

# 문장생성함수 (시작 텍스트, 생성 단어 개수)
def text_generation(sos, count):
    for _ in range(1, count):
        token_list = tokenizer.texts_to_sequences([sos])[0]
        token_list = pad_sequences([token_list],
                                   maxlen=max_len-1,
                                   padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=1) # 최대값 인덱스

        for word, idx in tokenizer.word_index.items():
            if idx == predicted:
                output = word
                break
        sos += " " + output
    return sos

# argmax 설명: 최대값의 인덱스 반환
data = [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2], [0.4, 0.3, 0.3]]
np.argmax([data], axis=-1)

print( text_generation("연애 하면서", 12) )

print( text_generation("꿀잼", 12) )

print( text_generation("최고의 영화", 12) )

print( text_generation("손발 이", 12) )