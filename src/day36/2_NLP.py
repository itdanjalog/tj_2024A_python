# 텐서플로 토크나이저
from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [
             '영실이는 나를 정말 정말 좋아해',
             '영실이는 영화를 좋아해'
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
print("단어 인덱스:",tokenizer.word_index)

# 인코딩된 결과
word_encoding = tokenizer.texts_to_sequences(sentences)
print( word_encoding )

# 사전에 없는 단어가 있을 때 인코딩 결과
new_sentences = ['영실이는 경록이와 나를 좋아해']
new_word_encoding = tokenizer.texts_to_sequences(new_sentences)
new_word_encoding

# 사전에 없는(Out Of Vocabulary) 단어 처리
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

new_word_encoding = tokenizer.texts_to_sequences(new_sentences)

print(word_index)
print(new_word_encoding)

# 단어사전 개수 설정
tokenizer = Tokenizer(num_words=3, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

new_word_encoding = tokenizer.texts_to_sequences(new_sentences)

print(word_index)
print(new_word_encoding)

# 문장의 길이 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(word_encoding)
print(padded)

# 패딩(뒤에 0 붙이기)
padded = pad_sequences(word_encoding, padding='post')
print(padded)

# 문장의 최대 길이 고정
padded = pad_sequences(word_encoding, padding='post',maxlen=4)
print(padded)

# 최대 길이보다 문장이 길 때 뒷부분 자르기
padded = pad_sequences(word_encoding, padding='post', truncating='post', maxlen=4)
print(padded)