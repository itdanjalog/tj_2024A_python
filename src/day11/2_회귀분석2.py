import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 간단한 가상의 학생 데이터 생성
data = pd.DataFrame({
    '공부시간': [2, 3, 5, 7, 1],
    '출석률': [80, 85, 90, 95, 70],
    '수학점수': [60, 65, 75, 85, 50]
})

# 회귀 공식 정의
Rformula = '수학점수 ~ 공부시간 + 출석률'

# 모델 피팅
model = smf.ols(Rformula, data=data).fit()

# 결과 출력
print(model.summary())

'''
 수학 점수(math_score)를 공부시간(study_hours)과 출석률(attendance_rate)로 예측하는 것입니다.
 
                  coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      6.7857      7.833      0.866      0.478     -26.916      40.487
공부시간           3.3929      0.443      7.651      0.017       1.485       5.301
출석률            0.5714      0.111      5.146      0.036       0.094       1.049

'''