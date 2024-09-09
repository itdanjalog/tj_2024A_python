'''
**MSE (Mean Squared Error, 평균 제곱 오차)**는 모델의 예측 성능을 평가하는 지표 중 하나입니다.
MSE는 예측 값과 실제 값 간의 차이를 제곱하여 평균을 구하는 방식으로 계산됩니다.

오차 계산:
각 데이터 포인트에 대해 예측 값과 실제 값의 차이를 계산합니다.
이 차이를 오차(error)라고 부릅니다.
예를 들어, 실제 값이 10이고 예측 값이 8이라면, 오차는  10−8=2입니다.

제곱:
각 오차를 제곱하여 양수로 변환합니다. 제곱을 하는 이유는 오차의 부호가 문제되지 않도록 하고, 큰 오차에 더 큰 페널티를 주기 위함입니다.
예를 들어, 오차가 2일 경우, 제곱 오차는 =4입니다.

평균 계산:
모든 제곱 오차의 평균을 구합니다. 이 평균이 바로 MSE입니다.
예를 들어, 3개의 데이터 포인트에서 제곱 오차가 4, 9, 1이라면, MSE는
4+9+1/3= 14/3 =4.67

특징과 장점
단위: MSE는 제곱된 오차의 평균이기 때문에, 예측 값과 실제 값의 단위의 제곱 단위로 표현됩니다. 예를 들어, 가격을 예측하는 모델에서 MSE의 단위는 제곱 원입니다.
민감성: MSE는 큰 오차에 더 큰 페널티를 부여합니다. 그래서 큰 오차가 있는 경우 MSE 값이 크게 증가합니다.
연속성: MSE는 연속적인 값을 가지며, 모델의 성능을 정량적으로 평가할 수 있습니다.

단점
해석의 어려움: MSE는 제곱 단위로 측정되므로 원래 데이터 단위와는 차이가 있어 해석이 어려울 수 있습니다.
큰 오차에 민감: 큰 오차에 대해 제곱하기 때문에, 아웃라이어(극단적인 값)에 민감합니다. 이로 인해 MSE가 매우 커질 수 있습니다.

'''
from sklearn.metrics import mean_squared_error

# 실제 값과 예측 값
Y_true = [3, -0.5, 2, 7]
Y_pred = [2.5, 0.0, 2, 8]

# MSE 계산
mse = mean_squared_error(Y_true, Y_pred)
print(f'MSE: {mse:.3f}') # MSE: 0.375

'''

'''

