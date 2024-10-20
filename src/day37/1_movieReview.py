# 필요 라이브러리 불러오기
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

import warnings
warnings.filterwarnings(action='ignore')

# Naver sentiment movie corpus v1.0 데이터 불러오기
train_file = tf.keras.utils.get_file(
    'ratings_train.txt', origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', extract=True)

train = pd.read_csv(train_file, sep='\t')

# 데이터 크기 및 샘플 확인
print("train shape: ", train.shape)
train.head()

# 레이블별 개수
cnt = train['label'].value_counts()
print(cnt)

# 레이블별 비율
sns.countplot(x='label',data=train)

# 결측치 확인
train.isnull().sum()

# 결측치(의견없음)가 특정 label값만 있는지 확인
train[train['document'].isnull()]

# 레이블 별 텍스트 길이
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
data_len=train[train['label']==1]['document'].str.len()
ax1.hist(data_len)
ax1.set_title('positive')

data_len=train[train['label']==0]['document'].str.len()
ax2.hist(data_len)
ax2.set_title('negative')
fig.suptitle('Number of characters')
plt.show()


# Mecab 형태소 설치
! git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git

    cd
    Mecab - ko -
    for -Google - Colab /

        ! bash
        install_mecab - ko_on_colab190912.sh

# Kkma, Komoran, Okt, Mecab 형태소
import konlpy
from konlpy.tag import Kkma, Komoran, Okt, Mecab

kkma = Kkma()
komoran = Komoran()
okt = Okt()
mecab = Mecab()

# 형태소별 샘플
text = "영실아안녕오늘날씨어때?"

def sample_ko_pos(text):
    print(f"==== {text} ====")
    print("kkma:",kkma.pos(text))
    print("komoran:",komoran.pos(text))
    print("okt:",okt.pos(text))
    print("mecab:",mecab.pos(text))
    print("\n")

sample_ko_pos(text)

text2 = "영실아안뇽오늘날씨어때?"
sample_ko_pos(text2)

text3 = "정말 재미있고 매력적인 영화에요 추천합니다."
sample_ko_pos(text3)

# 텍스트 전처리(영어와 한글만 남기고 삭제)
train['document'] = train['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
train['document'].head()

# 결측치 제거
train = train.dropna()
train.shape

# 스탑워드와 형태소 분석
def word_tokenization(text):
  stop_words = ["는", "을", "를", '이', '가', '의', '던', '고', '하', '다', '은', '에', '들', '지', '게', '도'] # 한글 불용어
  return [word for word in mecab.morphs(text) if word not in stop_words]

data = train['document'].apply((lambda x: word_tokenization(x)))
data.head()

# train과 validation 분할

training_size = 120000

# train 분할
train_sentences = data[:training_size]
valid_sentences = data[training_size:]

# label 분할
train_labels = train['label'][:training_size]
valid_labels = train['label'][training_size:]

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# vocab_size 설정
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
print("총 단어 갯수 : ",len(tokenizer.word_index))

# 5회 이상만 vocab_size에 포함
def get_vocab_size(threshold):
  cnt = 0
  for x in tokenizer.word_counts.values():
    if x >= threshold:
      cnt = cnt + 1
  return cnt

vocab_size = get_vocab_size(5) # 5회 이상 출현 단어
print("vocab_size: ", vocab_size)

oov_tok = "<OOV>" # 사전에 없는 단어
vocab_size = 15000

tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(data)
print(tokenizer.word_index)
print("단어 사전 개수:", len(tokenizer.word_counts))

# 문자를 숫자로 표현
print(train_sentences[:2])
train_sequences = tokenizer.texts_to_sequences(train_sentences)
valid_sequences = tokenizer.texts_to_sequences(valid_sentences)
print(train_sequences[:2])

# 문장의 최대 길이
max_length = max(len(x) for x in train_sequences)
print("문장 최대 길이:", max_length)

# 문장 길이를 동일하게 맞춘다
trunc_type='post'
padding_type='post'

train_padded = pad_sequences(train_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
valid_padded = pad_sequences(valid_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

train_labels = np.asarray(train_labels)
valid_labels = np.asarray(valid_labels)

print("샘플:", train_padded[:1])


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional

def create_model():
    model = Sequential([
                Embedding(vocab_size, 32),
                Bidirectional(LSTM(32, return_sequences=True)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# 가장 좋은 loss의 가중치 저장
checkpoint_path = 'best_performed_model.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                verbose=1)

# 학습조기종료
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# 학습
history = model.fit(train_padded, train_labels,
                validation_data=(valid_padded, valid_labels),
                callbacks=[early_stop, checkpoint], batch_size=64, epochs=10, verbose=2)

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

# 테스트 데이터 불러오기
test_file = tf.keras.utils.get_file(
    'ratings_test.txt', origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt', extract=True)

test = pd.read_csv(test_file, sep='\t')
test.head()

# 데이터 전처리
def preprocessing(df):
  df['document'] = df['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
  df = df.dropna()
  test_label = np.asarray(df['label'])
  test_data =  df['document'].apply((lambda x: word_tokenization(x)))
  test_data = tokenizer.texts_to_sequences(test_data)
  test_data = pad_sequences(test_data, truncating=trunc_type, padding=padding_type, maxlen=max_length)
  return test_data, test_label

test_data, test_label = preprocessing(test)
print(model.evaluate(test_data, test_label))

# 기본 모델 로드 후 평가
model2 = create_model()
model2.evaluate(test_data, test_label)

# 저장된 가중치 적용된 모델 로드 후 평가
model2.load_weights(checkpoint_path)
model2.evaluate(test_data, test_label)

!git clone https://github.com/SKTBrain/KoBERT.git

cd KoBERT

pip install -r requirements.txt


from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
tok_path = get_tokenizer()
sp  = SentencepieceTokenizer(tok_path)
print(sp('영실아오늘날씨어때?'))
print(sp('영실아 오늘 날씨 어때?'))


# 위와 동일한 전처리 과정
def word_tokenization_kobert(text):
    stop_words = ["는", "을", "를", '이', '가', '의', '던', '고', '하', '다', '은', '에', '들', '지', '게', '도'] # 한글 불용어
    return [word for word in sp(text) if word not in stop_words]

def train_preprocessing(df):
    df['document'] = df['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
    df = df.dropna()
    data =  df['document'].apply((lambda x: word_tokenization_kobert(x)))
    print(data.head())
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, truncating=trunc_type, padding=padding_type, maxlen=max_length)

    training_size = 120000
    train_sentences = data[:training_size]
    valid_sentences = data[training_size:]
    train_labels = np.asarray(df['label'][:training_size])
    valid_labels = np.asarray(df['label'][training_size:])
    return train_sentences, valid_sentences, train_labels, valid_labels

train_padded, valid_padded, train_labels, valid_labels = train_preprocessing(train)

model3 = create_model()
history3 = model3.fit(train_padded, train_labels,
                validation_data=(valid_padded, valid_labels),
                callbacks=[early_stop, checkpoint], batch_size=64, epochs=10, verbose=2)

plot_graphs(history3, 'accuracy')

plot_graphs(history3, 'loss')