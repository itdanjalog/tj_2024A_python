

import matplotlib.pyplot as plt
import statsmodels.api as sm

# 가상의 데이터 (키와 몸무게)
heights = [150, 160, 170, 180, 190]
weights = [50, 55, 65, 70, 80]


# 데이터 준비
X = sm.add_constant(heights)  # 상수항(절편)을 추가
model = sm.OLS(weights, X)     # OLS 회귀 모델 생성
results = model.fit()          # 모델 학습

# 예측값
predicted_weights = results.predict(X)

# 회귀 결과 출력
print(results.summary())
