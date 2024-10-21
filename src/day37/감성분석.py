import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. 데이터 로드
train_file = tf.keras.utils.get_file(
    'ratings_train.txt',
    origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', # 1:긍정(좋아요) # 0:부정(싫어요)
    extract=True
)
# 'ratings_train.txt': 다운로드할 파일의 이름을 지정합니다.
# origin: 파일을 다운로드할 URL을 지정합니다.
# extract=True: 만약 다운로드한 파일이 압축 파일이라면, 이 옵션은 파일을 자동으로 추출합니다.
# 이 단계에서는 리뷰 데이터셋이 포함된 ratings_train.txt 파일이 다운로드되고, train_file 변수에 해당 파일의 경로가 저장됩니다.



# 2. 데이터 로드 및 전처리
data = pd.read_csv(train_file, sep='\t')
data = data.dropna(how='any')  # 결측값 제거 # 값이 존재하지 않는 경우를 의미
# pd.read_csv(): Pandas를 사용하여 CSV 파일을 DataFrame 형식으로 읽어옵니다.
#
# train_file: 앞서 다운로드한 파일의 경로를 사용하여 데이터를 로드합니다.
# sep='\t': 이 파일은 탭으로 구분된 파일이므로, 구분자를 탭(\t)으로 설정합니다.
# data.dropna(how='any'): DataFrame에서 결측값이 있는 행을 제거합니다.
#
# how='any': 한 개라도 결측값이 존재하는 행을 모두 제거합니다. 즉, 모든 열이 결측값이 아닐 때만 해당 행이 남습니다.


# 리뷰와 레이블을 분리
texts = data['document'].values
labels = data['label'].values

# 3. 텍스트 토큰화 및 패딩
max_words = 20000  # 사용할 최대 단어 수
max_len = 100      # 최대 문장 길이
# max_words: 모델에서 사용할 최대 단어 수를 정의합니다. 여기서는 20,000개의 단어를 사용할 예정입니다.
# max_len: 모델에 입력할 최대 문장 길이를 설정합니다. 여기서는 100자로 제한하고 있습니다.


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len)
# Tokenizer(num_words=max_words): Keras의 Tokenizer 객체를 생성합니다. num_words 인자는 가장 빈번하게 등장하는 max_words개의 단어만 사용하도록 설정합니다.
# tokenizer.fit_on_texts(texts): 주어진 텍스트 데이터(texts)에 대해 토크나이저를 학습시킵니다. 이 과정에서 각 단어에 고유한 인덱스가 부여됩니다.
# tokenizer.texts_to_sequences(texts): 학습된 토크나이저를 사용하여 각 리뷰 텍스트를 숫자 시퀀스로 변환합니다. 각 단어는 해당 인덱스로 대체됩니다. 이 결과는 리뷰의 각 단어를 인덱스 번호로 나타내는 리스트의 리스트인 sequences에 저장됩니다.
# pad_sequences(sequences, maxlen=max_len): sequences 리스트의 각 시퀀스를 동일한 길이로 맞추기 위해 패딩을 적용합니다.
# maxlen=max_len: 각 시퀀스의 최대 길이를 max_len으로 설정하여, 이보다 긴 시퀀스는 잘리고, 짧은 시퀀스는 0으로 채워집니다. 이를 통해 모든 입력 데이터의 길이를 동일하게 만들어 모델에 입력할 수 있습니다.

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)


# 4. 모델 설계
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])
# Sequential: Keras의 Sequential 모델을 사용하여 층을 순차적으로 쌓아 모델을 생성합니다.
# Embedding(max_words, 128, input_length=max_len):
# Embedding: 단어 인덱스를 밀집 벡터(dense vector)로 변환합니다. 여기서는 각 단어를 128차원의 벡터로 표현합니다.
# max_words: 사용할 최대 단어 수로, 이 경우 20,000개의 단어가 포함됩니다.
# input_length=max_len: 입력 시퀀스의 길이를 지정합니다. 이 경우 100으로 설정합니다.
# LSTM(64, return_sequences=False):
# LSTM: 장기 기억을 통해 시퀀스 데이터를 처리하는 층입니다. 여기서는 64개의 LSTM 유닛을 사용합니다.
# return_sequences=False: 마지막 LSTM 셀의 출력을 반환합니다. 이 경우 최종 시퀀스 출력이 아니라 마지막 시간 단계의 출력을 사용합니다.
# Dense(1, activation='sigmoid'):
# Dense: 완전 연결 층입니다.
# 1: 출력 뉴런의 수로, 이진 분류이므로 1로 설정합니다.
# activation='sigmoid': 시그모이드 활성화 함수를 사용하여 0과 1 사이의 확률을 출력합니다.

# 5. 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer='adam': Adam 옵티마이저를 사용하여 모델의 가중치를 업데이트합니다. Adam은 일반적으로 잘 작동하는 옵티마이저입니다.
# loss='binary_crossentropy': 이진 분류 문제이므로 손실 함수로 이진 교차 엔트로피를 사용합니다.
# metrics=['accuracy']: 모델의 성능을 평가하기 위해 정확도를 지표로 설정합니다.

# 6. 모델 학습
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val))
# model.fit(): 모델을 학습시키는 메서드입니다.
# X_train: 훈련 데이터(입력 시퀀스).
# y_train: 훈련 데이터의 레이블(정답).
# epochs=5: 전체 데이터셋에 대해 5회 학습합니다.
# batch_size=128: 한 번의 업데이트에 사용할 샘플 수로, 여기서는 128로 설정합니다.
# validation_data=(X_val, y_val): 검증 데이터와 레이블을 제공하여 각 에포크 후 모델의 성능을 평가합니다.

# 7. 성능 평가
loss, accuracy = model.evaluate(X_val, y_val)
print(f"검증 데이터 정확도: {accuracy * 100:.2f}%")

# ------------------------------------------------------------------------------------------------------------- #
# 임의 리뷰 입력
new_reviews = [ "이 영화 정말 재미있었어요!", "정말 지루하고 재미없어요.", "보통이었어요. 그냥 그랬어요." , "맛없어." ]

# 입력 리뷰도 학습 데이터처럼 토큰화 및 패딩
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_len)

# 모델을 통해 감성 예측
predictions = model.predict(new_padded_sequences)

# 예측 결과 출력
for i, review in enumerate(new_reviews):
    sentiment = "긍정" if predictions[i] > 0.5 else "부정"
    print(f"리뷰: {review}")
    print(f"예측된 감성: {sentiment}, 확률: {predictions[i][0]:.4f}\n")

