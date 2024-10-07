import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 샘플 주행 시간과 이동 거리 데이터 생성 (노이즈 없음)
# 주행 시간 (hours)
x = np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9])
# 이동 거리 (km) = 60 * 주행 시간 (노이즈 없음)
y = np.array( [60, 120, 180, 240, 300, 360, 420, 480, 540 ])  # 자동차는 60km/h의 속도로 주행한다고 가정

print("주행 시간 (X):", x)
print("이동 거리 (Y):", y)

# 단순 선형 회귀 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# 모델 컴파일
model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# 모델 훈련
history = model.fit(x, y, epochs=500)

# 훈련 과정에서의 손실 시각화
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# 모델 평가
loss, mae = model.evaluate(x, y)
print(f'Loss: {loss:.4f}, MAE: {mae:.4f}')

# 예측
predicted_distance = model.predict(np.array([10]))  # 10시간 주행에 대한 이동 거리 예측
print(f'예측된 이동 거리 (주행 시간 10시간): {predicted_distance[0][0]:.2f} km')

# 데이터와 예측 결과 시각화
plt.scatter(x, y, label='Actual Data', color='blue')  # 실제 데이터
plt.scatter(10, predicted_distance, label='Predicted Distance', color='red')  # 예측값
plt.plot(x, model.predict(x), label='Regression Line', color='green')  # 회귀선
plt.xlabel('주행 시간 (시간)')
plt.ylabel('이동 거리 (km)')
plt.title('주행 시간과 이동 거리 관계')
plt.legend()
plt.show()