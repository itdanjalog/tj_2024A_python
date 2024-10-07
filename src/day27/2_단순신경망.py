
########## [2] p.64
import numpy as np

# 샘플 데이터셋 생성
x = np.arange(1, 6)

# y = 3x + 2
y = 3 * x + 2
print(x)
print(y)

import matplotlib.pyplot as plt

# 시각화
plt.plot(x, y)
plt.title('y = 3x + 2')
plt.show()

import tensorflow as tf
# 리스트형 # 시퀀셜
# tf.keras.Sequential: 신경망의 층을 순차적으로 쌓는 데 사용되는 Keras 모델입니다. 각 층은 이전 층의 출력을 입력으로 받습니다.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    # Dense(10): 10개의 노드를 가진 첫 번째 은닉층을 정의합니다. 이 층은 입력 데이터의 차원에 따라 자동으로 입력 노드를 설정합니다.
    tf.keras.layers.Dense(5),
    # Dense(5): 5개의 노드를 가진 두 번째 은닉층을 정의합니다. 이전 층(10개의 노드를 가진 층)에서의 출력을 입력으로 사용합니다.
    tf.keras.layers.Dense(1),
    # Dense(1): 1개의 노드를 가진 출력층을 정의합니다. 이 층은 모델의 최종 출력값을 제공합니다. 주로 회귀 문제에서 단일 값을 예측하는 데 사용됩니다.
])
# 첫 번째 은닉층: 10개의 노드 # 두 번째 은닉층: 5개의 노드 # 출력층: 1개의 노드
# 모델은 입력 데이터를 받아 연속적으로 층을 통과하면서 변환을 수행하며, 최종 출력은 1개의 값으로 반환됩니다. 이 구조는 주로 회귀 문제나 이진 분류 문제에 사용됩니다.

# add 함수로 레이어 추가
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(5))
model.add(tf.keras.layers.Dense(1))

model = tf.keras.Sequential([
    # 입력 데이터의 shape = (150, 4) 인 경우 input_shape 지정
    tf.keras.layers.Dense(10, input_shape=[4]),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Dense(1),
])

# 단순선형회귀 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# 모델 요약
model.summary()

# 긴 문자열 지정
# model.compile(optimizer='sgd', loss='mean_squared_error',
#               metrics=['mean_squared_error', 'mean_absolute_error'])
'''
optimizer='sgd': Stochastic Gradient Descent(확률적 경사 하강법) 옵티마이저를 사용합니다. 이는 일반적으로 모델 학습 시 사용되는 기본 옵티마이저입니다.
loss='mean_squared_error': 손실 함수로 평균 제곱 오차(MSE)를 사용합니다. 이는 예측값과 실제값 간의 차이를 제곱하여 평균한 값으로, 회귀 문제에서 자주 사용됩니다.
metrics=['mean_squared_error', 'mean_absolute_error']: 모델의 성능을 평가하기 위해 사용할 지표를 설정합니다. 여기서는 평균 제곱 오차(MSE)와 평균 절대 오차(MAE)를 사용합니다.
'''


# 짧은 문자열 지정
# model.compile(optimizer='sgd', loss='mse', metrics=['mse', 'mae'])

'''
이 부분은 위의 코드와 동일한 작업을 수행하지만, 손실 함수와 메트릭 이름을 짧은 형태로 지정했습니다.
loss='mse': 평균 제곱 오차의 약어를 사용했습니다.
metrics=['mse', 'mae']: 각 지표의 약어를 사용했습니다
'''

'''
두 코드 블록은 같은 의미를 가지며, 동일한 손실 함수와 메트릭을 설정합니다.
차이점은 첫 번째 코드에서는 함수 이름을 완전하게 적고, 두 번째 코드에서는 약어를 사용했다는 것입니다.
Keras에서는 이러한 약어를 사용하는 것이 일반적이며, 가독성을 높이는 데 도움이 됩니다.
'''

# TensorFlow 2.0부터는 학습률(learning rate)을 설정할 때 lr 대신 learning_rate 인자를 사용해야 합니다.
# 클래스 인스턴스 지정
#model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005),
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
#               loss=tf.keras.losses.MeanAbsoluteError(),
#               metrics=[tf.keras.metrics.MeanAbsoluteError(),
#                        tf.keras.metrics.MeanSquaredError()
#                        ])

# 컴파일
model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# 훈련
history = model.fit(x, y, epochs=1200)

import matplotlib.pyplot as plt

# 20 에포크까지 Loss 수렴에 대한 시각화
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['mae'], label='mae')
plt.xlim(-1, 20)
plt.title('Loss')
plt.legend()
plt.show()

# 검증
print( model.evaluate(x, y) )

# 예측
print( model.predict(np.array([10])) )
