# day15 > 3_와인통계분석
    # --> day12 > 5_와인통계분석 코드 복사해서 이어서 작성

################################## day 12 5_와인통계분석 ##########################
# 1. 데이터수집 : winequality-red.csv , winequality-white.csv
# 2. 데이터준비 :
# 2-1 : csv 파일의 열 구분자를 ;세미콜론 --> ,쉼표 변경 하여 csv 파일를 새로 만들기
import pandas as pd
red_pd = pd.read_csv( 'winequality-red.csv' , sep=';' , header= 0 , engine='python')
    # sep = 'csv구분자' # 기본값은 ,(쉼표)
    # header : = 0 첫번째 행을 열 이름으로 지정하겠다는 뜻
white_pd = pd.read_csv( 'winequality-white.csv' , sep=';' , header= 0 , engine='python' )
## 새로운 csv 만들기
red_pd.to_csv( 'winequality-red2.csv' , index=False)
    #  index=False : 데이터프레임 의 인덱스 열은 CSV 파일에 포함하지 않는다.
white_pd.to_csv( 'winequality-white2.csv' , index=False)

# 2-2 데이터 병합하기  , #  레드 와인 과 화이트 와인 분석하기 위해 하나로 합치기
print( red_pd.head() ) #  .head() : 데이터프레임 의 위에서부터 5개 행 출력
#  1. 열 추가  , # .insert( 삽입할열위치 , column='열이름' , value=값 ) 0번째(첫번째) 열에 type 열 이름으로 red 값들을 추가
red_pd.insert( 0 , column='type' , value='red' )
print( red_pd.head( ))
print( red_pd.shape ) # .shape  , (1599, 13)  # 행 개수 와 열 개수 반환
print( red_pd.shape[0] ) # .shape[0] : 행개수 , .shape[1] : 열 개수
# 2.
print( white_pd.head() )
white_pd.insert( 0 , column='type' , value='white' )
print( white_pd.head() )
print( white_pd.shape ) #  (4898, 13)
# 3. ㄷ데이터프레임 합치기 , pd.concat( [ 데이터프레임1 , 데이터프레임2 ] )
wine = pd.concat( [ red_pd , white_pd ] )
print( wine.shape ) # (6497, 13)
# 4. 합친 와인 데이터프레임 ---> csv 파일로 저장
wine.to_csv( 'wine.csv' , index=False )

# [3] 데이터 탐색
    # 1. 데이터프레임 의 기존정보 출력
print( wine.info() )
    # 2. 기술 통계
wine.columns = wine.columns.str.replace(' ' ,'_' ) # - 열이름의 공백이 있으면 _(밑줄) 변경
print( wine.head() )
    #   .describe() : 속성(열)마다 개수 , 평균 , std(표준편차) , 최소값 , 백분위수25% , 백분위수50% , 백분위수75% , 최대값
print( wine.describe() ) #
print( wine.describe()['quality']) # 와인의 등급 통계
print( wine.describe()['quality'].to_list() ) # 와인 등급의 통계 리스트
    #
print( wine['quality'].unique() )   # [5 6 7 4 8 3 9]
print( wine.quality.unique() )      # [5 6 7 4 8 3 9]
print( sorted( wine.quality.unique() ) )# 3 4 5 6 7 8 9  # 와인 등급의 중복 값 제거 하고 정렬
    #
print( wine['quality'].value_counts() ) # 특정한 열(등급) 별로 개수
print( wine['quality'].value_counts().to_list() ) # [2836, 2138, 1079, 216, 193, 30, 5]
print( wine['quality'].value_counts().to_json() ) # {"6":2836,"5":2138,"7":1079,"4":216,"8":193,"3":30,"9":5}
# [4] 데이터 모델링
    # 1. .groupby('그룹기준')['속성명']
    # type 속성으로 그룹 해서 quality 속성 기술 통계 구하기
print( wine.groupby('type')['quality'].describe() )
    # 2. type 속성으로 그룹 해서 quality 속성의 평균
print( wine.groupby('type')['quality'].mean() )
    # 3. type 속성으로 그룹 해서 quality 속성의 표준편차
print( wine.groupby('type')['quality'].std() )
    # 4. type 속성으로 그룹 해서 quality 속성의 평균,표준편차
print( wine.groupby('type')['quality'].agg([ 'mean','std']))

################################## ################### ##########################

#1. t-test
    # 원인변수( 독립변수 )  레드 , 화이트         ( 명목형 )
    # 결과변수( 종속변수 )  등급( 1, 2, 3, 4, 5 )( 연속형 )
# [1] 모듈 호출
from scipy import stats
# [2] 데이터 수집 : 두 집단 표본 만들기
    # 판다스데이터프레임
    # wine( 레드와인 과 화이트와인 자료를 합친 데이터프레임 객체 )
    # .loc( 조건 , 출력할 열 )
# print( wine.loc[ wine['type'] == 'red' ,'quality'] )
레드와인등급집단 = wine.loc[ wine['type'] == 'red'    , 'quality'] # type열의 값이 red 이면 등급 출력
화이트와인등급집단 = wine.loc[ wine['type'] == 'white' , 'quality' ] # type열의 값이 white 이면 등급 출력
# [3] t-test 독립 표본 검증
t통계량 , p값 = stats.ttest_ind( 레드와인등급집단 , 화이트와인등급집단 )
# [4] 결론 확인
print( t통계량 )      # -9.685649554187696 # 첫번째 집단의 평균이 두번째 집단보다 낮다는 뜻 # 해석 : 평균적으로  화이트 등급이 더 높다.
print( p값 )         # 4.888069044201508e-22 #  e-소수부크기
if p값 < 0.05 :   print( '해당 가설은 유의미 하다 ')
else :     print( '해당 가설은 무의미 하다. ')

# 2. 회귀분석 ( 다중 선형 회귀분석 )
# [1] 모듈 호출
from statsmodels.formula.api import ols  # [ statsmodels ] 모듈 설치
# [2] 회귀모형수식( 종속변수와 독립변수를 구성하는 방식/공식 : 종속변수명 ~ 독립변수1 + 독립변수2 + 독립변수3 )
회귀모형수식 = 'quality ~ alcohol' # 종속변수와 독립변수를 정의 #
    # 가설 : 알콜 수치에 따라 등급 확인
    # 독립변수(원인/연속성) 알콜
    # 종속변수(결과/연속성) 등급

# [3] ols(선형 회귀 모델)  # ols( 회귀모형수식 , data = 표본집단 )
선형회귀모델 = ols( 회귀모형수식 , data = wine )
선형회귀모델결과 = 선형회귀모델.fit()
print( 선형회귀모델결과.summary( ) )

'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:       (종속변수)quality   R-squared:                       0.197
Model:                            OLS   Adj. R-squared:                  0.197
Method:                 Least Squares   F-statistic:                     1598.
Date:                Tue, 03 Sep 2024   Prob (F-statistic):          1.50e-312
Time:                        13:07:03   Log-Likelihood:                -7623.4
No. Observations:                6497   AIC:                         1.525e+04
Df Residuals:                    6495   BIC:                         1.526e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                    coef        std err       t         P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept(절편)      2.4053      0.086     27.988      0.000       2.237       2.574
alcohol(독립변수)     0.3253      0.008     39.970      0.000       0.309       0.341
==============================================================================
Omnibus:                      123.922   Durbin-Watson:                   1.636
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              258.800
Skew:                           0.031   Prob(JB):                     6.34e-57
Kurtosis:                       3.976   Cond. No.                         94.3
==============================================================================


Intercept(절편)  : 독립변수가 0 일때 종속변수의 예측값 # X가 0일때 Y의 값
인과 관계 : 알콜 과 와인 등급 관계 
                          coef        std err       t         P>|t|      [0.025      0.975]
    - alcohol(독립변수)     0.3253      0.008     39.970      0.000       0.309       0.341
    
        - coef : 회귀계수 
            0.3253 : 알콜 과 와인 등급 관계 해석 # 알콜(독립변수)이 1 단위 증가할때마다 등급(종복변수)이 0.3253 증가 예측 . ( 알콜이 3 증가하면 등급은 대략 1증가한다 예측)
        - std err : 표준 오차 
        - t : t검증 통계량  
        - P>|t| : t검증 p값 
            -  0.000 : 0.05 보다 작으면 검증 의 효과가 있다.
        - [0.025      0.975] : 앞뒤 2.5% 버리고 사이에 존재하면 신뢰할수 있는 구간 뜻 한다.
       
'''











