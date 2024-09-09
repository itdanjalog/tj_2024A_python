'''
1. MSE (Mean Squared Error, 평균 제곱 오차)
정의:
MSE는 실제 값과 예측 값 간의 차이의 제곱 평균을 의미합니다.
오차를 제곱하여 양수로 만든 후 평균을 구하는 방식으로, 오차가 클수록 더 큰 영향을 줍니다.
MSE가 작을수록 모델의 성능이 더 좋다는 것을 의미합니다.
'''
from sklearn.metrics import mean_squared_error

# 실제 값과 예측 값 예시
Y_true = [3, -0.5, 2, 7]
Y_pred = [2.5, 0.0, 2, 8]

# MSE 계산
mse = mean_squared_error(Y_true, Y_pred)
print(f'MSE: {mse:.3f}') # MSE: 0.375

'''
2. RMSE (Root Mean Squared Error, 루트 평균 제곱 오차)
정의:
RMSE는 MSE의 제곱근을 취한 값으로, 실제 값과 예측 값 간의 평균 오차의 크기를 원래 단위로 변환해줍니다.
RMSE는 실제 값과 예측 값 간의 평균적인 오차 크기를 해석하기 쉽게 만들어줍니다.
'''
import numpy as np

# MSE가 구해진 후 RMSE 계산
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.3f}') # RMSE: 0.612

'''
3. R² (R-Squared, 결정 계수)
정의:
R²는 모델이 전체 변동성을 얼마나 잘 설명하는지 나타냅니다.
0에서 1 사이의 값으로, 1에 가까울수록 예측 성능이 좋다는 의미입니다.
R² 값이 1이면 모델이 완벽하게 데이터를 설명하고, 0이면 모델이 전혀 설명하지 못한다는 의미입니다.
'''

from sklearn.metrics import r2_score

# R² 계산
r2 = r2_score(Y_true, Y_pred) # R²: 0.949
print(f'R²: {r2:.3f}')


'''
4. Y 절편 (Intercept)
Y 절편은 선형 회귀식에서 종속 변수의 값이 모든 독립 변수가 0일 때의 값입니다.
회귀 방정식에서 y = b0 + b1x1 + b2x2 + ... + bnxn 중에서 b0가 Y 절편입니다.

'''
from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 훈련
model = LinearRegression()
model.fit([[1], [2], [3]], [2, 4, 6])

# Y 절편 출력
print(f'Y 절편 (Intercept): {model.intercept_:.3f}')
'''
5. 회귀 계수 (Regression Coefficients) 리그레션 코에피션츠
회귀 계수는 각 독립 변수(특성)가 종속 변수에 미치는 영향력을 나타냅니다.
회귀 방정식에서 y = b0 + b1x1 + b2x2 + ... + bnxn 중에서 b1, b2, ..., bn가 회귀 계수입니다.
계수 값이 클수록 해당 변수의 영향력이 더 크다는 것을 의미합니다.

계수란?
수학/통계학에서의 계수: 특정 수식에서 변수와 곱해지는 상수입니다. 예를 들어, 수식 3x에서 3이 계수입니다.
회귀 분석에서의 회귀 계수: 독립 변수의 변화가 종속 변수에 미치는 영향을 나타내는 값입니다. 예를 들어, 선형 회귀 모델에서 회귀 계수는 각 독립 변수의 중요성과 영향을 측정합니다.
'''
# 선형 회귀 모델의 회귀 계수 출력
print(f'회귀 계수 (Coefficients): {model.coef_}')

'''
MSE: 예측 값과 실제 값 사이의 오차 크기를 제곱하여 계산한 평균.
RMSE: MSE의 제곱근, 해석 가능한 실제 값의 단위로 변환된 오차 크기.
R²: 모델의 설명력, 1에 가까울수록 모델이 데이터를 잘 설명함.
Y 절편: 모든 독립 변수가 0일 때의 종속 변수 값.
회귀 계수: 각 독립 변수가 종속 변수에 미치는 영향력.
'''





















