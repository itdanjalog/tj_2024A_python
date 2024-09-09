
# 10장. 회귀분석 (1) 주택가격 회귀 분석
'''
1.데이터 로드 및 전처리: 데이터를 로드하고, 결측치를 처리하거나 이상치를 제거하는 과정.
2.변수 선택: 독립 변수와 종속 변수를 설정.
3.데이터 분할: 학습용 데이터와 테스트용 데이터로 나누기.
4.모델 학습: 회귀 모델을 훈련시키는 과정.
5.모델 평가: 테스트 데이터를 통해 모델의 성능을 평가.
6.예측: 학습된 모델로 새로운 데이터에 대한 예측을 수행.

'''
# 향후 버전 업에 대한 경고 메시지 출력 안하기
import warnings
warnings.filterwarnings(action='ignore')
# 특정 패키지에서 경고 메시지가 많이 발생하지만, 코드 실행 결과에는 영향을 미치지 않는 경우에 자주 사용합니다.

# ======================================== ======================================== #

### - 머신러닝 패키지 sklearn 설치 !pip install sklearn

## 1) 데이터 수집

## ** 보스턴주택가격 데이터셋은 인종차별문제가 제기되어, sklearn version 1.2. 이후로 제공하지 않음. **
### - from sklearn.datasets import load_boston 대신에 원본데이터셋을 다운로드 받아서 사용하는 것으로 수정함.

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"

raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# sep="\s+" # 구분자로 하나 이상의 공백을 사용한다는 뜻입니다. 이 데이터는 공백으로 구분된 형식입니다.
      # \s는 정규 표현식에서 공백 문자를 의미합니다. # +는 정규 표현식에서 바로 앞에 오는 패턴이 하나 이상 반복되는 것을 의미
#  skiprows=22 # 상단 22줄을 건너뛰고 데이터를 읽겠다는 의미입니다. 보스턴 데이터셋의 첫 22줄에는 메타데이터가 포함되어 있어서 이를 생략합니다.
# 데이터셋에 별도의 열 제목(header)이 없기 때문에 None으로 설정했습니다. 이후에 열 이름을 별도로 지정해줄 수 있습니다.
print( raw_df )
#C:\Users\MSI\Desktop\tj_2024A_python\src\day19\1_주택가격분석.py:21: SyntaxWarning: invalid escape sequence '\s'
#  raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# 특성(feature) 데이터와 타겟(target) 데이터로 나누는 과정
첫번째 = raw_df.values[::2, :]
두번째 = raw_df.values[1::2, :2]

data = np.hstack( [ 첫번째 , 두번째 ]) # 수평으로 배열을 쌓는 함수입니다.
print( data )
target = raw_df.values[1::2, 2]
'''
data: 짝수 번째 행의 모든 데이터와 홀수 번째 행의 첫 번째와 두 번째 열을 결합한 특성 데이터.
target: 홀수 번째 행에서 세 번째 열을 추출한 타겟(종속 변수) 데이터.
'''

# X 피처 => 주택 관련 변수들
print(data.shape)

# 타깃 피처 => 주택가격
print(target.shape)

# ======================================== ======================================== #
## 2) 데이터 준비 및 탐색
# **독립 변수(특성)**들의 이름
feature_names = ['CRIM', 'ZN', 'INDUS','CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
'''
CRIM: 범죄율
ZN: 25,000 평방 피트 이상의 주택 구역 비율
INDUS: 비소매 상업 지역 비율
CHAS: 찰스 강 인접 여부 (1: 인접, 0: 인접하지 않음)
NOX: 일산화질소 농도
RM: 주택당 방의 수
AGE: 1940년 이전에 건축된 주택 비율
DIS: 5개의 보스턴 고용 중심지까지의 거리
RAD: 고속도로 접근성 지수
TAX: 재산세율
PTRATIO: 학생-교사 비율
B: 아프리카계 미국인의 비율
LSTAT: 인구 중 저소득층의 비율
'''
boston_df = pd.DataFrame(data, columns = feature_names)
print( boston_df.head() )

# boston_df['PRICE'] = target: 타겟 데이터(주택 가격)를 PRICE라는 이름의 열로 추가합니다.
boston_df['PRICE'] = target

print( boston_df.head() )

# boston_df.shape: 데이터셋의 **형상(크기)**를 반환합니다. 여기서는 (행의 개수, 열의 개수) 형식으로 출력됩니다.
print('보스톤 주택 가격 데이터셋 크기 : ', boston_df.shape)
print( boston_df.info() )

# ## 3) 분석 모델 구축
# 평가 지표를 계산하고 회귀 계수를 확인하는 전형적인 머신러닝 워크플로우입니다.

from sklearn.linear_model import LinearRegression     # scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
'''
LinearRegression: 선형 회귀 모델을 생성하는 클래스입니다.
train_test_split: 데이터셋을 훈련용과 테스트용으로 나누는 함수입니다.
mean_squared_error, r2_score: 모델 평가를 위한 지표를 계산하는 함수들로, 각각 **평균 제곱 오차(MSE)**와 **R²(결정 계수)**를 계산합니다.
'''


# X, Y 분할하기
Y = boston_df['PRICE']
X = boston_df.drop(['PRICE'], axis=1, inplace=False)
'''
Y = boston_df['PRICE']: Y는 타겟 변수(종속 변수)인 **주택 가격(PRICE)**입니다. 이 열만 따로 가져옵니다.
X = boston_df.drop(['PRICE'], axis=1, inplace=False): X는 독립 변수(특성)들입니다. PRICE 열을 제외한 모든 열을 선택하여 새로운 DataFrame을 만듭니다.
'''

# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=156)
'''
train_test_split: 데이터를 **훈련용(70%)**과 **테스트용(30%)**으로 분할합니다.
test_size=0.3: 테스트 데이터는 전체의 30%.
random_state=156: 랜덤 분할 시 재현성을 보장하기 위한 시드 값입니다.
결과적으로, 훈련 데이터 (X_train, Y_train)과 테스트 데이터 (X_test, Y_test)로 나눕니다.
'''
# 선형회귀분석 : 모델 생성
lr = LinearRegression()
# 선형회귀분석 : 모델 훈련
lr.fit(X_train, Y_train)
'''
lr = LinearRegression(): 선형 회귀 모델을 생성합니다.
lr.fit(X_train, Y_train): 훈련 데이터를 사용하여 모델을 학습시킵니다. 이 과정에서 모델이 회귀 계수와 절편 값을 학습합니다.
'''

# 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr.predict(X_test)
'''
lr.predict(X_test): 학습된 모델을 사용하여 테스트 데이터에 대한 주택 가격을 예측하고, 그 결과를 **Y_predict**에 저장합니다.
'''

# ## 4) 결과 분석 및 시각화
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
'''
mean_squared_error(Y_test, Y_predict): **평균 제곱 오차(MSE)**를 계산합니다. MSE는 실제 값과 예측 값의 차이의 제곱의 평균으로, 작을수록 모델의 성능이 좋다는 의미입니다.
np.sqrt(mse): **루트 평균 제곱 오차(RMSE)**를 계산하여 오차의 크기를 해석 가능한 단위로 변환합니다.
r2_score(Y_test, Y_predict): **결정 계수(R²)**를 계산합니다. 이는 모델이 데이터를 얼마나 잘 설명하는지를 나타내는 지표로, 1에 가까울수록 설명력이 좋습니다.

MSE와 RMSE가 작을수록 모델의 예측 오차가 적다는 것을 의미합니다.
R²는 1에 가까울수록 좋습니다. 여기서는 약 0.757로, 예측이 어느 정도 정확하다는 것을 보여줍니다.

'''



print('Y 절편 값: ', lr.intercept_)
print('회귀 계수 값: ', np.round(lr.coef_, 1))

'''
lr.intercept_: 모델이 학습한 Y 절편 값을 출력합니다.
lr.coef_: 각 특성에 대한 회귀 계수를 출력합니다. 회귀 계수는 각 특성이 종속 변수에 미치는 영향을 나타내며, 값이 클수록 중요한 특성입니다.
'''

coef = pd.Series(data = np.round(lr.coef_, 2), index=X.columns)
print( coef.sort_values(ascending = False) )
'''
pd.Series(): 회귀 계수를 pandas Series로 변환합니다.
np.round(lr.coef_, 2): 회귀 계수를 소수점 둘째 자리까지 반올림합니다.
coef.sort_values(ascending=False): 회귀 계수를 내림차순으로 정렬하여 가장 큰 영향력을 가진 특성부터 순서대로 출력합니다.

'''


'''
## - 회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=5)

x_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='PRICE', data=boston_df, ax=axs[row][col])

plt.show()
'''

