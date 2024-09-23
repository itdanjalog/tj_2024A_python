# 분류분석1 : 로지스틱 분석
# 데이터셋
# import pandas as pd
# data = pd.read_csv('https://bit.ly/wine_csv_data')
## 1.데이터셋 설명 ##
# 데이터셋에는 알코올 도수, 당도, pH 값이 들어있고
# 네 번째 열 class는 타깃값인데 0이면 레드 와인, 1이면 화이트 와인이다
## 2. 훈련용(8) : 테스트용(2) 분류 한다.
## 3. 모델 구축과 테스트 진행하여 결과를 아래와 같이 출력하시오.
'''
첫번째 테스트의 레드 와인 확률 : [0.28128279 0.71871721] , 레드확률:0.28128278745944046, 화이트확률:0.7187172125405595
첫번째 데스트의 와인 예측 : 1.0
예측 정확도 : 0.7876923076923077
'''












































import pandas as pd
from sklearn.metrics import accuracy_score

# 데이터 수집
data = pd.read_csv('https://bit.ly/wine_csv_data')
# 데이터셋에는 알코올 도수, 당도, pH 값이 들어있고
# 네 번째 열 class는 타깃값인데 0이면 레드 와인, 1이면 화이트 와인이다.
# 레드 와인인지 화이트 와인인지 분류하는 것은 이진 분류이고 화이트 와인이 양성 클래스이다. 다시 말해, 이 문제는 전체 와인 데이터에서 화이트 와인을 골라내는 문제이다.

# 로지스틱분석
X = data[ ['alcohol' ,'sugar' , 'pH'] ] # 스케일링(표준화) 한 데이터 독립변수
Y = data[ 'class' ] # 종속변수
# 훈련용 데이터 , 평가용 데이터 분할
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split( X , Y , test_size=0.2 , random_state=0 )


# 데이터 스케일링 (표준화)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler( )
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# 모델 객체 생성
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(  random_state=45)
# 훈련용 데이터를 피팅(훈련) 하기
model.fit( x_train , y_train )
# 예측 하기 # 평가용으로 수행
y_predict = model.predict( x_test ) # 이진 분류 예측
result  = model.predict_proba( x_test )
print( f'첫번째 테스트의 레드 와인 확률 : {result[0] } , 레드확률:{ result[0][0] }, 화이트확률:{result[0][1]}' ) # [0.27468642 0.72531358]
print( f'첫번째 데스트의 와인 예측 : {y_predict[0]}' ) # 와인 예측 : 1.0 # 화이트확인
print( f'예측 정확도 : { accuracy_score( y_test , y_predict ) }' ) # 예측 정확도 확인  # 0.7876923076923077
