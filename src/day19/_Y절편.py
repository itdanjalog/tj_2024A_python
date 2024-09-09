'''
**Y 절편 (Y-Intercept)**는 회귀 분석에서 직선의 Y축과 교차하는 지점을 나타내는 값입니다.
이 값은 회귀 모델에서 종속 변수의 예측 값이 독립 변수의 값이 0일 때의 값을 의미합니다.

정의
Y 절편은 회귀 직선의 방정식에서 독립 변수 X가 0일 때 종속 변수 Y의 값입니다.
회귀 직선의 일반적인 방정식은 다음과 같습니다.
Y = P + P1X
P : 절편 P1회귀계수

'''
from sklearn.linear_model import LinearRegression

# 데이터 준비
X = [[1], [2], [3], [4], [5]]  # 2D 배열 형태로 변환
Y = [2, 4, 5, 4, 5]

# LinearRegression 모델 생성
model = LinearRegression()

# 모델 훈련
model.fit(X, Y)

# Y 절편과 회귀 계수 출력
print(f'Y 절편 (Intercept): {model.intercept_:.2f}') # Y 절편 (Intercept): 2.20
print(f'회귀 계수 (Coefficient): {model.coef_[0]:.2f}') # 회귀 계수 (Coefficient): 0.60

# 예측 값 계산
test_X = [[6], [7]]  # 2D 배열 형태로 변환
predictions = model.predict(test_X)
print(f'예측 값 for X = {test_X}: {predictions}') # 예측 값 for X = [[6], [7]]: [5.8 6.4]