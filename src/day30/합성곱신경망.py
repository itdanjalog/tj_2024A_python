import tensorflow as tf
import numpy as np

# MNIST 손글씨 이미지 데이터 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)

# 새로운 출력값 배열을 생성 (홀수: 1, 짝수: 0)
y_train_odd = []
for y in y_train:
    if y % 2==0:
        y_train_odd.append(0)
    else:
        y_train_odd.append(1)

y_train_odd = np.array(y_train_odd)
y_train_odd.shape

print(y_train[:10])
print(y_train_odd[:10])

# Validation 데이터셋 처리 # 밸-리-데이-션
y_valid_odd = []
for y in y_valid:
    if y % 2==0:
        y_valid_odd.append(0)
    else:
        y_valid_odd.append(1)

y_valid_odd = np.array(y_valid_odd)
y_valid_odd.shape

# 정규화(Normalization) # 밸-릿
x_train = x_train / 255.0
x_valid = x_valid / 255.0

# 채널 추가
x_train_in = tf.expand_dims(x_train, -1)
x_valid_in = tf.expand_dims(x_valid, -1)

print(x_train_in.shape, x_valid_in.shape)


# Functional API를 사용하여 모델 생성

inputs = tf.keras.layers.Input(shape=(28, 28, 1))
# Input 레이어는 모델의 입력을 정의합니다.
# 이 모델은 28x28 크기의 흑백 이미지(채널 수 1)를 입력으로 받습니다. (예: MNIST 데이터셋)

conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
# Conv2D는 합성곱 레이어로, 입력 이미지에서 특징을 추출합니다.
# 32개의 필터를 사용하여 3x3 크기의 커널을 적용하고, ReLU 활성화 함수로 비선형성을 추가합니다.
# 출력의 형태는 (28-3+1, 28-3+1) = 26x26으로, 각 필터가 26x26 크기의 특성 맵을 생성합니다. (채널은 32)
pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)
# MaxPooling2D는 다운샘플링 레이어로, 특성 맵의 크기를 줄여 계산량을 줄이고 중요한 특징을 강조합니다.
# 여기서는 2x2 풀링 윈도우를 사용하여 출력 크기를 절반으로 줄입니다.
# 출력 형태는 **(13, 13, 32)**가 됩니다.
flat = tf.keras.layers.Flatten()(pool)
# Flatten은 다차원 배열인 특성 맵을 1차원 벡터로 변환합니다.
# (13, 13, 32) 크기의 특성 맵을 1차원 벡터 (13 * 13 * 32 = 5408)로 변환합니다.

flat_inputs = tf.keras.layers.Flatten()(inputs)
# 원본 28x28 이미지를 직접 평평하게 만들어 1차원 벡터로 변환합니다.
# 출력 벡터의 크기는 28 * 28 = 784입니다.
concat = tf.keras.layers.Concatenate()([flat, flat_inputs])
# Concatenate는 두 개의 1차원 벡터를 결합합니다.
# 합성곱된 특성 맵과 원본 이미지에서 직접 추출된 특성 벡터를 결합합니다.
# 결합된 출력의 크기는 5408 + 784 = 6192입니다.
outputs = tf.keras.layers.Dense(10, activation='softmax')(concat)
# Dense는 완전 연결층으로, 10개의 노드를 가지고 있습니다. 각 노드는 10개의 클래스 중 하나에 해당하며, 이는 분류 작업을 수행합니다.
# Softmax 활성화 함수를 사용하여 출력이 각 클래스에 대한 확률이 되도록 합니다.

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
# 입력과 출력을 정의하여 모델을 생성합니다.
# model.summary()는 모델의 구조를 출력하여 각 레이어의 출력 형태와 파라미터 수를 보여줍니다.
model.summary()
# 입력과 출력을 정의하여 모델을 생성합니다.
# model.summary()는 모델의 구조를 출력하여 각 레이어의 출력 형태와 파라미터 수를 보여줍니다.

# 이 모델은 28x28 크기의 흑백 이미지를 입력으로 받아, 합성곱 신경망을 통해 추출한 특성 맵과 원본 이미지의 정보를 결합하여 10개의 클래스로 분류하는 모델입니다.


# 모델 구조 출력 및 이미지 파일로 저장
# # 모델 구조 출력 및 이미지 파일로 저장
# from tensorflow.keras.utils import plot_model
# plot_model(model, show_shapes=True, show_layer_names=True, to_file='functional_cnn.png')



# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(x_train_in, y_train,
                    validation_data=(x_valid_in, y_valid),
                    epochs=10)

# 모델 성능
val_loss, val_acc = model.evaluate(x_valid_in, y_valid)
print(val_loss, val_acc)



# Functional API를 사용하여 모델 생성

inputs = tf.keras.layers.Input(shape=(28, 28, 1), name='inputs')

conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_layer')(inputs)
pool = tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_layer')(conv)
flat = tf.keras.layers.Flatten(name='flatten_layer')(pool)

flat_inputs = tf.keras.layers.Flatten()(inputs)
concat = tf.keras.layers.Concatenate()([flat, flat_inputs])
digit_outputs = tf.keras.layers.Dense(10, activation='softmax', name='digit_dense')(concat)

odd_outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='odd_dense')(flat_inputs)

model = tf.keras.models.Model(inputs=inputs, outputs=[digit_outputs, odd_outputs])

model.summary()

# 모델의 입력과 출력을 나타내는 텐서
print(model.input)
print(model.output)
#
# plot_model(model, show_shapes=True, show_layer_names=True, to_file='multi_output_cnn.png')


# 모델 컴파일
model.compile(optimizer='adam',
              loss={'digit_dense': 'sparse_categorical_crossentropy', 'odd_dense': 'binary_crossentropy'},
              loss_weights={'digit_dense': 1, 'odd_dense': 0.5}, # loss = 1.0 *sparse_categorical_crossentropy + 0.5*binary_crossentropy
              metrics=['accuracy', 'accuracy'])

# 모델 훈련
history = model.fit({'inputs': x_train_in}, {'digit_dense': y_train, 'odd_dense': y_train_odd},
                    validation_data=({'inputs': x_valid_in},  {'digit_dense': y_valid, 'odd_dense': y_valid_odd}),
                    epochs=10)

# 모델 성능
model.evaluate({'inputs': x_valid_in}, {'digit_dense': y_valid, 'odd_dense': y_valid_odd})

# 샘플 이미지 출력
import matplotlib.pylab as plt

def plot_image(data, idx):
    plt.figure(figsize=(5, 5))
    plt.imshow(data[idx])
    plt.axis("off")
    plt.show()

plot_image(x_valid, 0)


digit_preds, odd_preds = model.predict(x_valid_in)
print(digit_preds[0])
print(odd_preds[0])

digit_labels = np.argmax(digit_preds, axis=-1)
digit_labels[0:10]

odd_labels = (odd_preds > 0.5).astype(np.int_).reshape(1, -1)[0]
odd_labels[0:10]

# 앞의 모델에서 flatten_layer 출력을 추출
base_model_output = model.get_layer('flatten_layer').output

# 앞의 출력을 출력으로 하는 모델 정의
base_model = tf.keras.models.Model(inputs=model.input, outputs=base_model_output, name='base')
base_model.summary()

# plot_model(base_model, show_shapes=True, show_layer_names=True, to_file='base_model.png')

# Sequential API 적용
digit_model = tf.keras.Sequential([
                                   base_model,
                                   tf.keras.layers.Dense(10, activation='softmax'),
                                   ])
digit_model.summary()

# plot_model(digit_model, show_shapes=True, show_layer_names=True, to_file='digit_model.png')

# 모델 컴파일
digit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = digit_model.fit(x_train_in, y_train,
                    validation_data=(x_valid_in, y_valid),
                    epochs=10)

# 베이스 모델의 가중치를 고정 (Freeze Model)

base_model_frozen = tf.keras.models.Model(inputs=model.input, outputs=base_model_output, name='base_frozen')
base_model_frozen.trainable = False
base_model_frozen.summary()

# Functional API 적용
dense_output = tf.keras.layers.Dense(10, activation='softmax')(base_model_frozen.output)
digit_model_frozen = tf.keras.models.Model(inputs=base_model_frozen.input, outputs=dense_output)
digit_model_frozen.summary()

# 모델 컴파일
digit_model_frozen.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = digit_model_frozen.fit(x_train_in, y_train,
                    validation_data=(x_valid_in, y_valid),
                    epochs=10)

# 베이스 모델의 Conv2D 레이어의 가중치만 고정 (Freeze Layer)
base_model_frozen2 = tf.keras.models.Model(inputs=model.input, outputs=base_model_output, name='base_frozen2')
base_model_frozen2.get_layer('conv2d_layer').trainable = False
base_model_frozen2.summary()

# Functional API 적용
dense_output2 = tf.keras.layers.Dense(10, activation='softmax')(base_model_frozen2.output)
digit_model_frozen2 = tf.keras.models.Model(inputs=base_model_frozen2.input, outputs=dense_output2)
digit_model_frozen2.summary()

# 모델 컴파일
digit_model_frozen2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = digit_model_frozen2.fit(x_train_in, y_train,
                    validation_data=(x_valid_in, y_valid),
                    epochs=10)
