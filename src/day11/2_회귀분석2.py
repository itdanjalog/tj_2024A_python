import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 간단한 가상의 학생 데이터 생성
data = pd.DataFrame({
    'study_hours': [2, 3, 5, 7, 1],
    'attendance_rate': [80, 85, 90, 95, 70],
    'math_score': [60, 65, 75, 85, 50]
})

# 회귀 공식 정의
Rformula = 'math_score ~ study_hours + attendance_rate'

# 모델 피팅
model = smf.ols(Rformula, data=data).fit()

# 결과 출력
print(model.summary())