# 기본 라이브러리
import pandas as pd
import numpy as np
import tensorflow as tf

# 네이버 영화리뷰 데이터셋 불러오기
# tf.keras.utils.get_file: 텐서플로우 유틸리티를 통해 영화 리뷰 데이터를 다운로드.
file = tf.keras.utils.get_file(
    'ratings_train.txt',
    origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',
    extract=True)
# pandas.read_csv: 데이터를 탭으로 구분하여 읽어들임.
df = pd.read_csv(file, sep='\t')

# 데이터 임의샘플 확인
print( df[1000:1007] )

# 형태소 분석기
# !pip install konlpy
# konlpy: 한국어 텍스트 처리를 위해 사용하는 라이브러리. 이 중 Okt 형태소 분석기를 사용하여 텍스트를 분리.
from konlpy.tag import Okt
okt = Okt()

# 데이터 전처리
# word_tokenization: 주어진 문장을 형태소로 분리하여 리스트로 반환.
def word_tokenization(text):
  return [word for word in okt.morphs(text)]

# 결측치를 제거하고, 1000개의 데이터를 추출한 후, 불필요한 특수문자를 제거하여 텍스트를 정리.
def preprocessing(df):
  df = df.dropna()
  df = df[1000:2000]  # 샘플 데이터 1000개, 학습시간을 줄이고자 함
  df['document'] = df['document'].str.replace("[^A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ ]","" ,  regex=True ) # pd판다스버전 2.x이상부터 , regex=True 속성)
  data =  df['document'].apply((lambda x: word_tokenization(x)))
  return data

# 텍스트 데이터 1000개 전처리 후 불러오기
review = preprocessing(df)
len(review)

# 형태소 분리된 데이터 확인
print(review[:10])

# 토큰화 및 패딩
# Tokenizer: 텍스트를 숫자로 변환하는 도구.
# pad_sequences: 시퀀스의 길이를 동일하게 맞추기 위해 패딩을 추가.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()


def get_tokens(review):
    # tokenizer.word_index에 각 단어와 그에 대응하는 인덱스가 저장됩니다
    tokenizer.fit_on_texts(review)
    # tokenizer.word_index에 저장된 단어의 총 개수를 계산한 후, 1을 더해 총 단어 수(total_words)를 구합니다.
    # 1을 더하는 이유는 인덱스가 0부터 시작하지 않고 1부터 시작하도록 하기 위함입니다.
    total_words = len(tokenizer.word_index)+1
    # review의 각 문장을 단어 인덱스의 리스트로 변환합니다.
    # 예를 들어, 문장 ['이 영화는 정말 재미있다']는 [3, 7, 12, 45, 78]과 같은 숫자 리스트로 변환됩니다.
    # 각 숫자는 단어의 고유 인덱스입니다.
    tokenized_sentences = tokenizer.texts_to_sequences(review)
    input_sequences = []
    for token in tokenized_sentences:
        for t in range(1, len(token)):
            # token[:t+1]은 토큰 리스트에서 처음부터 t+1번째 단어까지를 슬라이싱한 시퀀스입니다.
            #  [3, 7, 12, 45, 78]에서 첫 번째 반복에서는 [3, 7],
            #  두 번째 반복에서는 [3, 7, 12],
            #  세 번째 반복에서는 [3, 7, 12, 45]와 같이 점점 길어지는 시퀀스가 만들어집니다.
            n_gram_sequence = token[:t+1]
            # 생성된 n-그램 시퀀스를 input_sequences 리스트에 추가합니다.
            input_sequences.append(n_gram_sequence)
    # 최종적으로 생성된 n-그램 시퀀스 리스트(input_sequences)와 단어 총 개수(total_words)를 반환합니다.
    return input_sequences, total_words

input_sequences, total_words = get_tokens(review)
print( input_sequences[31:40] ) # n_gram으로 리스트된 데이터샘플 확인

# 단어 사전
print("감동 ==>> ",tokenizer.word_index['감동'])
print("영화 ==>> ",tokenizer.word_index['영화'])
print("코믹 ==>> ",tokenizer.word_index['코믹'])


# 문장의 길이 동일하게 맞추기
# 설명: LSTM 모델에서는 모든 입력 시퀀스의 길이가 동일해야 하므로, 시퀀스의 길이를 맞춰줍니다. 여기서 pad_sequences를 사용해 짧은 시퀀스의 앞부분을 0으로 채워 길이를 동일하게 만듭니다.
# max([len(word) for word in input_sequences]): 모든 시퀀스에서 가장 긴 시퀀스의 길이를 찾습니다.
# pad_sequences(input_sequences, maxlen=max_len, padding='pre'): 모든 시퀀스를 최대 길이(max_len)에 맞춰 앞쪽에 0을 추가하여 길이를 동일하게 맞춥니다. padding='pre'는 앞쪽에 패딩을 추가하는 옵션입니다.
max_len = max([len(word) for word in input_sequences])
print("max_len:", max_len)
input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_len, padding='pre'))

# 입력텍스트와 타겟
from tensorflow.keras.utils import to_categorical
# 설명: X는 입력 데이터로, 각 시퀀스의 마지막 단어를 제외한 부분입니다. 즉, 시퀀스의 처음부터 마지막 단어 직전까지가 입력으로 사용됩니다.
X = input_sequences[:,:-1]  # 마지막 값은 제외함
# y는 타겟 데이터로, 각 시퀀스의 마지막 단어를 one-hot encoding하여 변환한 값입니다. 즉, 예측해야 할 단어입니다. to_categorical 함수를 사용하여 각 단어를 one-hot 벡터로 변환합니다.
y = to_categorical(input_sequences[:,-1],num_classes=total_words) # 마지막 값만 이진 클래스 벡터로 변환

# y를 설명하기 위한 예시
# 설명: to_categorical 함수는 주어진 숫자를 one-hot encoding 벡터로 변환합니다.
# [0, 1, 2, 3]라는 숫자 배열이 주어지면, 각 숫자는 길이가 4인 벡터로 변환됩니다. 각 벡터는 해당 인덱스의 위치에 1을 넣고 나머지는 0으로 채워집니다.
a = to_categorical([0, 1, 2, 3], num_classes=4)
print( a )
# [0, 1, 2, 3]가 각각 [1. 0. 0. 0.], [0. 1. 0. 0.], [0. 0. 1. 0.], [0. 0. 0. 1.]로 변환된 것을 확인할 수 있습니다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

embedding_dim = 256

model = Sequential([
    # Embedding 층
    # 설명: 임베딩 층은 단어를 밀집 벡터(dense vector)로 변환하는 역할을 합니다. 각 단어는 고유한 숫자로 인덱싱되어 있으며, 이 숫자를 해당 단어에 대응하는 벡터로 변환합니다.
    # input_dim=total_words: 입력으로 받을 단어의 총 개수(사전 크기)를 지정합니다. 즉, 모델이 다룰 수 있는 단어의 범위입니다.
    # output_dim=embedding_dim: 각 단어가 밀집 벡터로 변환된 후의 벡터 차원입니다. 예를 들어, embedding_dim=256이면 각 단어는 256차원의 벡터로 표현됩니다.
    # input_length=max_len-1: 입력 데이터의 시퀀스 길이를 지정합니다. 여기서 max_len-1은 마지막 단어를 제외한 나머지 시퀀스의 길이를 의미합니다.
    Embedding(input_dim=total_words,
              output_dim=embedding_dim,
              input_length=max_len-1),
    # 설명: 양방향 LSTM(Bidirectional LSTM) 레이어입니다. LSTM은 시퀀스 데이터를 처리하기 위한 순환 신경망(RNN) 구조 중 하나입니다. 양방향 LSTM은 입력 시퀀스를 정방향과 역방향으로 모두 처리하여 더 많은 정보를 학습할 수 있습니다.
    # units=256: LSTM 층의 유닛 수를 256개로 설정합니다. 이는 LSTM이 학습할 파라미터의 수를 의미합니다.
    Bidirectional(LSTM(units=256)),
    # 설명: 출력층입니다. total_words는 모델이 예측할 수 있는 단어의 총 개수로, 각 단어에 대한 확률 분포를 반환합니다.
    Dense(units=total_words, activation='softmax'),
])

# loss='categorical_crossentropy': 다중 클래스 분류 문제이므로 손실 함수로 categorical_crossentropy를 사용합니다. 이 함수는 모델이 예측한 확률 분포와 실제 타겟(one-hot encoding 된 값) 간의 차이를 계산합니다.
# optimizer='adam': Adam 옵티마이저를 사용합니다. Adam은 학습률을 자동으로 조정하며, 일반적으로 좋은 성능을 내는 옵티마이저입니다.
# metrics=['accuracy']: 학습 동안 정확도를 모니터링합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model.fit(X, y, epochs=20, verbose=1) # 20

# 문장생성함수 (시작 텍스트, 생성 단어 개수)
def text_generation(sos, count):
    for _ in range(1, count):
        token_list = tokenizer.texts_to_sequences([sos])[0]
        token_list = pad_sequences([token_list],
                                   maxlen=max_len-1,
                                   padding='pre')
        # 모델에 token_list를 넣고 예측을 수행하면, 다음에 나올 단어에 대한 예측 확률 분포를 얻게 돼요. 이때 모델이 예측한 확률들 중 가장 높은 확률을 가진 단어의 인덱스를 np.argmax()로 찾습니다.
        predicted = np.argmax(model.predict(token_list), axis=1) # 최대값 인덱스

        # for 루프에서 word_index.items()를 순회하면서 predicted에 해당하는 단어를 찾아냅니다. 그 단어를 output으로 저장하고, 그 단어를 이어서 sos에 붙여 나가는 방식으로 문장을 만들어 나갑니다.
        for word, idx in tokenizer.word_index.items():
            if idx == predicted:
                output = word
                break
        sos += " " + output
    return sos

# argmax 설명: 최대값의 인덱스 반환
data = [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2], [0.4, 0.3, 0.3]]
print( np.argmax([data], axis=-1) )

print( text_generation("연애 하면서", 12) )

print( text_generation("꿀잼", 12) )

print( text_generation("최고의 영화", 12) )

print( text_generation("손발 이", 12) )