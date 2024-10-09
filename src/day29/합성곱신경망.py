# tensorflow 모듈 import
import tensorflow as tf

# MNIST 손글씨 이미지 데이터 로드
# MNIST는 0부터 9까지의 손글씨 숫자 이미지로 구성된 데이터셋입니다.
mnist = tf.keras.datasets.mnist
# x_train과 x_valid는 이미지 데이터, y_train과 y_valid는 해당 이미지의 레이블(숫자)을 나타냅니다.
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
print( x_train[0 , : , : ] )
print( y_train[0] )

print(x_train.shape, y_train.shape)
#(60000, 28, 28) : 3차원 텐서 / 3차원 리스트/배열
# (60000,) : 1차원 텐서 / 1차원 리스트/배열
print(x_valid.shape, y_valid.shape) #(10000, 28, 28) (10000,)

# 샘플 이미지 출력
import matplotlib.pylab as plt

def plot_image(data, idx):
    plt.figure(figsize=(5, 5))
    plt.imshow(data[idx], cmap="gray")
    plt.axis("off")
    plt.show()

plot_image(x_train, 0)

print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

# 정규화(Normalization)
# 데이터 정규화를 수행합니다. 각 픽셀 값을 255.0으로 나누어 0과 1 사이의 값으로 변환합니다. 정규화는 모델 학습을 더 빠르고 안정적으로 만들어줍니다.
x_train = x_train / 255.0
x_valid = x_valid / 255.0

print( x_train[0 , : , : ] )
print( y_train[0] )

print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

# 채널 추가
print(x_train.shape, x_valid.shape) # (60000, 28, 28) 및 (10000, 28, 28)일 것입니다.

# 새로운 차원을 추가하여 (28, 28) 이미지를 (28, 28, 1)
x_train_in = x_train[..., tf.newaxis]
# 여기서 1은 채널 수를 나타내며, 흑백 이미지이므로 1입니다.
x_valid_in = x_valid[..., tf.newaxis]
# CNN은 일반적으로 3차원 입력을 기대합니다. 이 입력은 (높이, 너비, 채널) 형식으로 구성됩니다.
# 합성곱 신경망이 입력 데이터를 이해하고 처리할 수 있도록 하기 위한 필수적인 단계입니다

print(x_train_in.shape, x_valid_in.shape)

####################################################################
# Sequential API를 사용하여 샘플 모델 생성

#Sequential API를 사용하여 샘플 모델 생성
model = tf.keras.Sequential([
    # 리스트형식으로 순차적으로 층을 추가하는 방법입니다.

    # Convolution 적용 (32 filters)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv'),
        #  32개의 필터를 가진 3x3 크기의 합성곱 층을 추가합니다
        #  activation='relu'는 ReLU 활성화 함수를 사용하며,
        #  , 입력 모양은 (28, 28, 1)로 지정합니다.
    # Max Pooling 적용
    tf.keras.layers.MaxPooling2D((2, 2), name='pool'),
        #  2x2 크기의 최대 풀링 층을 추가하여 특성 맵의 크기를 줄입니다.
    # Classifier 출력층
    tf.keras.layers.Flatten(),
        #  다차원 배열을 1차원 배열로 변환합니다.
    tf.keras.layers.Dense(10, activation='softmax'),
        # 10개의 클래스를 출력하는 완전 연결(Dense) 층을 추가하며, softmax 활성화 함수를 사용하여 각 클래스에 대한 확률을 제공합니다.
])
#  **레이어(Layer)**는 신경망의 기본 구성 요소입니다.
# 합성곱 레이어, 풀링 레이어, 그리고 Flatten 레이어는 모두 **은닉 레이어(hidden layer)**에 속합니다.


# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # optimizer: Adam 옵티마이저를 사용합니다.
    # loss: 손실 함수로 sparse_categorical_crossentropy를 사용합니다. 이는 레이블이 정수형일 때 사용하는 손실 함수입니다.
    # metrics: 평가 지표로 정확도(accuracy)를 사용합니다.

# 모델 훈련
history = model.fit(x_train_in, y_train,
                    # x_train_in, y_train: 훈련 데이터와 레이블입니다.
                    validation_data=(x_valid_in, y_valid),
                    # 검증 데이터와 레이블로, 각 에포크마다 모델의 성능을 검증합니다.
                    epochs=10)
                    # 전체 데이터셋을 10회 반복하여 훈련합니다.

# 모델 훈련 과정에서 기록된 손실과 정확도 데이터를 포함한 객체입니다.

print( model.evaluate(x_valid_in, y_valid) ) #  손실 값과 정확도를 반환합니다.

def plot_loss_acc(history, epoch):

    loss, val_loss = history.history['loss'], history.history['val_loss']
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    # loss: 훈련 손실 값.
    # val_loss: 검증 손실 값.
    # acc: 훈련 정확도.
    # val_acc: 검증 정확도.


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # plt.subplots(1, 2, figsize=(12, 4))는 1행 2열로 구성된 서브플롯을 생성합니다. figsize는 전체 그래프의 크기를 지정합니다.


    axes[0].plot(range(1, epoch + 1), loss, label='Training')
    axes[0].plot(range(1, epoch + 1), val_loss, label='Validation')
    axes[0].legend(loc='best')
    axes[0].set_title('Loss')
    # axes[0]는 첫 번째 서브플롯(손실 그래프)에 대한 객체입니다.
    # plot 메소드를 사용하여 훈련 손실과 검증 손실을 그래프에 그립니다.
    # legend 메소드를 호출하여 그래프의 범례를 추가합니다.
    # set_title 메소드를 사용하여 그래프의 제목을 'Loss'로 설정합니다.

    axes[1].plot(range(1, epoch + 1), acc, label='Training')
    axes[1].plot(range(1, epoch + 1), val_acc, label='Validation')
    axes[1].legend(loc='best')
    axes[1].set_title('Accuracy')
    # axes[1]는 두 번째 서브플롯(정확도 그래프)에 대한 객체입니다.
    # 훈련 정확도와 검증 정확도를 그래프에 그립니다.
    # 범례와 제목을 추가하여 그래프를 구성합니다.

    plt.show()

plot_loss_acc(history, 10)

# 모델 구조
print( model.summary() )

# 입력 텐서 형태 #
# 모델의 입력 텐서 형태를 출력합니다. 입력 데이터의 크기와 형식을 보여주며, CNN의 경우 보통 (높이, 너비, 채널 수) 형식입니다.
print( model.inputs )

# 출력 텐서 형태
# 설명: 모델의 출력 텐서 형태를 출력합니다. 모델이 어떤 형태의 출력을 생성하는지를 나타내며, 일반적으로 클래스의 수에 해당하는 크기를 가집니다.
print( model.outputs )

# 레이어
# 설명: 모델에 포함된 모든 레이어의 리스트를 출력합니다. 각 레이어의 정보(이름, 타입 등)를 확인할 수 있습니다.
print( model.layers )

# 첫번째 레이어 선택
# 설명: 모델의 첫 번째 레이어를 선택하여 출력합니다. 이 레이어의 상세 정보를 확인할 수 있습니다.
print( model.layers[0] )

# 첫번째 레이어 입력
# 설명: 첫 번째 레이어에 대한 입력 텐서를 출력합니다. 이 입력은 모델의 입력 데이터에 해당합니다.
print( model.layers[0].input )

# 첫번째 레이어 출력
# 설명: 첫 번째 레이어의 출력 텐서를 출력합니다. 이 출력은 레이어가 처리한 후의 데이터 형태를 나타냅니다.
print( model.layers[0].output )

# 첫번째 레이어 가중치
# 설명: 첫 번째 레이어의 모든 가중치를 출력합니다. 이 가중치는 신경망의 학습 과정에서 조정됩니다.
print( model.layers[0].weights )

# 첫번째 레이어 커널 가중치
# 설명: 첫 번째 레이어의 커널(또는 필터) 가중치를 출력합니다. 이 가중치는 합성곱 연산에 사용됩니다.
print( model.layers[0].kernel )

# 첫번째 레이어 bias 가중치
# 설명: 첫 번째 레이어의 bias 가중치를 출력합니다. 이 가중치는 활성화 함수에 더해져 최종 출력을 계산하는 데 사용됩니다.
print( model.layers[0].bias )

# 레이어 이름 사용하여 레이어 선택
# 설명: 이름을 사용하여 특정 레이어를 선택합니다. 여기서는 'conv'라는 이름을 가진 레이어를 찾고 그 정보를 출력합니다. 이를 통해 코드에서 레이어를 직접 참조할 수 있어 유용합니다.
print( model.get_layer('conv') )

# 샘플 이미지의 레이어별 출력을 리스트에 추가 (첫번째, 두번째 레이어)
    # tf.keras.Model을 사용하여 새로운 모델을 정의합니다. 이 모델은 기존의 model에서 입력과 출력을 새롭게 지정합니다.
activator = tf.keras.Model(inputs=model.inputs,
                           # 기존 모델의 입력을 사용합니다. 즉, 모델에 들어오는 데이터의 형태를 그대로 유지합니다.
                           outputs=[layer.output for layer in model.layers[:2]] )
                            # 기존 모델의 첫 두 레이어의 출력을 출력으로 지정합니다. 즉, 첫 번째 레이어의 출력과 두 번째 레이어의 출력을 반환하는 새로운 모델이 생성됩니다.
                            # 여기서 model.layers[:2]는 모델의 첫 번째와 두 번째 레이어를 선택합니다.
                            # 각 레이어의 출력을 가져오기 위해 리스트 컴프리헨션을 사용합니다.
activations =activator.predict(x_train_in[0][tf.newaxis, ...])
# x_train_in[0]는 훈련 데이터에서 첫 번째 이미지를 선택합니다.
# [tf.newaxis, ...]를 추가하여 이 데이터를 새로운 차원을 추가해 배치 형태로 만듭니다. 즉, 1개의 샘플로 모델에 입력할 수 있도록 변환합니다.
# 이렇게 하면 입력 데이터의 형태가 (1, 28, 28, 1)이 되어, 모델이 요구하는 형식에 맞춰집니다.

print( len(activations) )


# 이 코드는 Keras 모델의 합성곱 레이어와 풀링 레이어에서 추출된 특징을 시각적으로 나타내어, 모델이 입력 이미지에서 어떤 정보를 추출하고 있는지 이해할 수 있도록 돕습니다.
# 각 레이어의 출력은 모델이 학습한 패턴이나 특성을 나타내며, 이를 통해 모델의 내부 동작을 더 잘 이해할 수 있습니다.

# 첫 번째 레이어(conv) 출력층
conv_activation = activations[0]
print( conv_activation.shape )

# Convolution 시각화
fig, axes = plt.subplots(4, 8)
fig.set_size_inches(10, 5)

for i in range(32):
# 첫 번째 합성곱 레이어에서 사용한 필터(또는 커널)의 수가 32개
    axes[i//8, i%8].matshow(conv_activation[0, :, :, i], cmap='viridis')
    axes[i//8, i%8].set_title('kernel %s'%str(i), fontsize=10)
    plt.setp( axes[i//8, i%8].get_xticklabels(), visible=False)
    plt.setp( axes[i//8, i%8].get_yticklabels(), visible=False)

plt.tight_layout()
plt.show()


# 두 번째 레이어(pool) 출력층
pooling_activation = activations[1]
print(pooling_activation.shape)

# 시각화
fig, axes = plt.subplots(4, 8)
fig.set_size_inches(10, 5)

for i in range(32):
    axes[i//8, i%8].matshow(pooling_activation[0, :, :, i], cmap='viridis')
    axes[i//8, i%8].set_title('kernel %s'%str(i), fontsize=10)
    plt.setp( axes[i//8, i%8].get_xticklabels(), visible=False)
    plt.setp( axes[i//8, i%8].get_yticklabels(), visible=False)

plt.tight_layout()
plt.show()

# 32개의 커널(필터)은 이어지는 것이 아니라, 각 커널이 독립적으로 작용하여 입력 데이터에 대해 병렬로 합성곱 연산을 수행합니다
# 32개의 커널을 사용하면, 각각의 커널에 의해 32개의 특징 맵이 생성

# 풀링은 화질을 저하시킬 수 있지만, 네트워크의 효율성을 높이고 일반화 능력을 향상시키는 데 중요한 역할을 합니다.
# 적절한 풀링 전략을 선택하는 것이 모델의 성능을 최적화하는 데 중요합니다.
# 화질 저하가 우려된다면, 풀링의 크기와 유형을 조절하거나 대안을 고려하는 것이 좋습니다.
