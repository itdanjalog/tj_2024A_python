# from gluonnlp.data import SentencepieceTokenizer # gluonnlp
# from kobert.utils import get_tokenizer

import numpy as np
import pandas as pd
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

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
