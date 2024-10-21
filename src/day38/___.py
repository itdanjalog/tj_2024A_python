import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 1. 데이터 준비 (예시 질문과 답변 쌍)
questions = [
    "안녕하세요",
    "오늘 날씨 어때요?",
    "내일은 몇 도에요?",
    "지금 몇 시에요?",
    "잘 지내고 계신가요?"
]

answers = [
    "안녕하세요! 무엇을 도와드릴까요?",
    "오늘은 맑고 기온이 25도입니다.",
    "내일은 20도 정도 될 것 같아요.",
    "지금은 3시 30분입니다.",
    "네, 잘 지내고 있어요!"
]

# 2. 텍스트 전처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# 단어와 정수 인덱스 맵 생성
word_index = tokenizer.word_index

# 질문과 답변을 정수 인덱스로 변환
encoder_input_sequences = tokenizer.texts_to_sequences(questions)
decoder_input_sequences = tokenizer.texts_to_sequences(answers)

# 패딩
max_sequence_length = max(max(len(seq) for seq in encoder_input_sequences),
                          max(len(seq) for seq in decoder_input_sequences))
encoder_input_sequences = pad_sequences(encoder_input_sequences, maxlen=max_sequence_length)
decoder_input_sequences = pad_sequences(decoder_input_sequences, maxlen=max_sequence_length)

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(encoder_input_sequences, decoder_input_sequences, test_size=0.2,
                                                  random_state=42)

# 3. 모델 정의
latent_dim = 256

# 인코더
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(input_dim=len(word_index) + 1, output_dim=latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# 디코더
decoder_inputs = Input(shape=(max_sequence_length,))
decoder_embedding = Embedding(input_dim=len(word_index) + 1, output_dim=latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델 컴파일
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. 모델 학습
y_train_reshaped = np.expand_dims(y_train, -1)  # 디코더 출력을 맞추기 위해 차원 변경
model.fit([X_train, y_train], y_train_reshaped, epochs=100, batch_size=32,
          validation_data=([X_val, y_val], np.expand_dims(y_val, -1)))


# 5. 예측
def predict_response(input_text):
    # 입력 텍스트 전처리
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)

    # 인코더 상태 생성
    states_value = encoder_lstm.predict(encoder_embedding(input_sequence))

    # 디코더 초기 입력
    target_sequence = np.zeros((1, max_sequence_length))
    target_sequence[0, 0] = word_index.get('안녕하세요', 1)  # 시작 토큰으로 '안녕하세요' 사용

    # 예측 결과 저장
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_lstm.predict([target_sequence] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word.get(sampled_token_index, '')

        # 종료 조건: max_length 또는 stop_token
        if sampled_char == '' or len(decoded_sentence) > max_sequence_length:
            stop_condition = True
            continue

        decoded_sentence += ' ' + sampled_char

        # 예측 결과 업데이트
        target_sequence[0, len(decoded_sentence)] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()


# 챗봇과 대화
user_input = "오늘 날씨 어때요?"
response = predict_response(user_input)
print(f"사용자: {user_input}")
print(f"챗봇: {response}")
