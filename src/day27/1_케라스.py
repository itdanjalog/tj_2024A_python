

########## [1] p.56 경사하강법 (Gradient Descent)
import numpy as np
import matplotlib.pyplot as plt

# 샘플에 활용할 데이터 셋 만들기
def make_linear(w=0.5, b=0.8, size=50, noise=1.0):
    # w: 선형 방정식의 기울기 (default: 0.5).
    # b: 선형 방정식의 y-절편 (default: 0.8).
    # size: 생성할 데이터 포인트의 수 (default: 50).
    # noise: 데이터에 추가할 잡음의 크기 (default: 1.0).
    x = np.random.rand(size) #  0과 1 사이의 균일 분포에서 size 개수의 난수로 이루어진 배열 x를 생성합니다.
    y = w * x + b # y: 선형 방정식   x에 대한 목표값을 계산합니다.

    # noise: 주어진 범위에서 균일하게 잡음을 생성합니다. -noise와 noise 사이의 값이 랜덤하게 생성됩니다.
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)

    yy = y + noise # yy: 원래의 y 값에 잡음을 추가하여 실제 관측값을 생성합니다.

    plt.figure(figsize=(10, 7)) # 그래프의 크기를 설정합니다.

    plt.plot(x, y, color='r', label=f'y = {w}*x + {b}') # 선으로 그래프에 표시합니다.
    plt.scatter(x, yy, label='data') # 생성된 데이터 포인트 yy를 산점도로 그래프에 표시합니다.
    plt.legend(fontsize=20)
    plt.show()
    print(f'w: {w}, b: {b}') # 기울기 w와 절편 b 값을 출력합니다.
    return x, yy # 생성된 x 값과 잡음이 추가된 yy 값을 반환합니다.

x, y = make_linear(w=0.3, b=0.5, size=100, noise=0.01)

# 이 함수는 주어진 매개변수에 따라 선형 데이터를 생성하고, 잡음을 추가한 후 이를 시각화합니다.
# 마지막으로 생성된 x 값과 잡음이 추가된 y 값을 반환하여, 나중에 분석이나 모델링에 사용할 수 있도록 합니다.

# 위의 호출은 기울기 0.3, 절편 0.5 데이터 포인트 100개, 잡음 크기  0.01로 선형 데이터를 생성하고 시각화합니다.


######################### w (기울기)와  b (절편)를 업데이트하여 손실(오차)를 최소화하는 방법을 보여줍니다.
# 최대 반복 횟수 # num_epoch: 최대 반복 횟수 (여기서는 1000회).
num_epoch = 1000

# 학습율 (learning_rate)
learning_rate = 0.005 # learning_rate: 파라미터 업데이트의 크기를 조절하는 학습율 (0.005).

# 에러 기록
errors = [] # errors: 각 반복(epoch)에서의 손실(오차)을 기록할 리스트.

# random 한 값으로 w, b를 초기화 합니다.
w = np.random.uniform(low=0.0, high=1.0) # **w**와 b: 기울기와 절편을 0과 1 사이의 랜덤 값으로 초기화합니다.
b = np.random.uniform(low=0.0, high=1.0) # **w**와 b: 기울기와 절편을 0과 1 사이의 랜덤 값으로 초기화합니다.

for epoch in range(num_epoch): # 지정된 최대 반복 횟수만큼 학습을 수행합니다.
    # Hypothesis 정의
    y_hat = w * x + b # 현재 기울기 w와 절편 b를 사용 하여 입력x에 대한 예측값을 계산한다.

    # Loss Function 정의
    error = 0.5 * ((y_hat - y) ** 2).sum()
    # error: 예측값과 실제값의 차이를 제곱하여 평균한 값으로, 손실 함수를 계산합니다. 여기서  0.5를 곱하는 것은 미분 시 계산 편의를 위해서입니다.

    if error < 0.005: # 손실이 특정 값 (0.005)보다 작아지면 학습을 조기 종료합니다.
        break

    # Gradient(기울기) 미분 계산
    # w: 기울기를 업데이트합니다. 예측 오차에 x를 곱한 값의 합에 학습율을 곱하여 현재의 w에서 빼줍니다.
    w = w - learning_rate * ((y_hat - y) * x).sum()
    # 절편을 업데이트합니다. 예측 오차의 합에 학습율을 곱하여 현재의 b에서 빼줍니다.
    b = b - learning_rate * (y_hat - y).sum()


    # 현재 손실을 기록하고, 매 5회마다 현재의 에포크, w, b, 손실을 출력합니다.
    errors.append(error)
    if epoch % 5 == 0:
        print("{0:2} w = {1:.5f}, b = {2:.5f} error = {3:.5f}".format(epoch, w, b, error))

print("----" * 15)
print("{0:2} w = {1:.1f}, b = {2:.1f} error = {3:.5f}".format(epoch, w, b, error))


plt.figure(figsize=(10, 7))
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# 이 코드는 선형 회귀 모델을 학습하여 주어진 데이터  x와 실제값  y에 대해 예측값
# yhat 을 개선하는 과정을 설명합니다. 학습은 경사하강법을 사용하며, 손실이 적어질수록
# w와  b가 최적화됩니다. 최종적으로 손실의 변화를 시각적으로 확인할 수 있습니다.