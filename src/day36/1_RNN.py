# 라이브러리 불러오기
import tensorflow as tf

# 임베딩 레이어
embedding_layer = tf.keras.layers.Embedding(100, 3)
result = embedding_layer(tf.constant([12,8,15, 20])) #더미 데이터 입력
print(result)

# 임베딩 레이어 활용
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(100, 3, input_length=32))
model.add(tf.keras.layers.LSTM(units=32))
model.add(tf.keras.layers.Dense(units=1))
model.summary()

# 라이브러리 불러오기
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 임베딩 레이어 활용

model = Sequential()
model.add(Embedding(100, 3, input_length=32))
model.add(LSTM(32))
model.add(Dense(1))
model.summary()

# Bidirectional LSTM
from tensorflow.keras.layers import Bidirectional

model = Sequential()
model.add(Embedding(100,3))
model.add(Bidirectional(LSTM(32))) # 양방향 RNN
model.add(Dense(1))
model.summary()

# 스태킹RNN 예제
model = Sequential()
model.add(Embedding(100,32))
model.add(LSTM(32, return_sequences=True))  # 전체 시퀀스 출력 (batch_size, timesteps, units)
model.add(LSTM(32))
model.add(Dense(1))
model.summary()


# 순환 드룹아웃
model = Sequential()
model.add(Embedding(100, 32))
model.add(LSTM(32, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()