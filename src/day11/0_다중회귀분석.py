import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 가상의 데이터 생성
np.random.seed(0)  # 결과 재현성을 위해 시드 설정

data = {
    '성별': np.random.choice(['male', 'female'], size=100),
    '스포츠': np.random.randint(1, 11, size=100),  # 1부터 10까지의 정수
    '드라마': np.random.randint(1, 11, size=100)   # 1부터 10까지의 정수
}

df = pd.DataFrame(data)

# 다중회귀 분석
# R 스타일 공식: 'drama_preference ~ gender + sports_preference'
# 여기서 gender는 더미 변수로 자동 변환됩니다.
model = smf.ols('드라마 ~ 성별 + 스포츠', data=df).fit()

# 결과 출력
print(model.summary())