# 분류분석2 : 결정트리 분석
# 데이터셋
# import pandas as pd
# data = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')

## 1.데이터셋 설명 ##
# 데이터셋에는 무게 , 길이 , 대각선길이,높이,너비 값이 들어있고
# 첫 번째 열 Species 는 타깃값 인데 어종 이다.
# 어종은 'Bream' , 'Roach' , 'Whitefish' , 'Parkki' , 'Perch' , 'Pike' , 'Smelt' 분류 한다.

## 2. 테이터를 훈련용(7) : 테스트용(3) 분류 한다.

## 3. 결정트리 모델 구축과 테스트 하여 정확도를 하고 시각화 하시오
### (출력) 개선 전 결정 트리모델 정확도 : 0.625
### (시각화)

## 4. 최적의 하이퍼 파라미터를 찾으시오.
### params = { 'max_depth' : [  2 , 6 , 10 , 14  ],  'min_samples_split' : [ 2 , 4 , 6 , 8  ] }
### (출력) 평균 정확도:0.7747035573122529, 최적 파라미터:{'max_depth': 10, 'min_samples_split': 4 }

## 5. 결정트리 모델 구축과 테스트 하여 정확도를 하고 시각화 하시오
### (출력) 개선 후 결정트리모델 정확도 : 0.6458333333333334
### (시각화)






























import pandas as pd
from sklearn.metrics import accuracy_score

# 데이터 수집
data = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')
# 데이터셋에는 알코올 도수, 당도, pH 값이 들어있고
# 네 번째 열 class는 타깃값인데 0이면 레드 와인, 1이면 화이트 와인이다.
# 레드 와인인지 화이트 와인인지 분류하는 것은 이진 분류이고 화이트 와인이 양성 클래스이다. 다시 말해, 이 문제는 전체 와인 데이터에서 화이트 와인을 골라내는 문제이다.
feature_label = ['Weight' , 'Length' , 'Diagonal' , 'Height' , 'Width']
target_label = [ 'Bream' , 'Roach' , 'Whitefish' , 'Parkki' , 'Perch' , 'Pike' , 'Smelt']

# 로지스틱분석
X = data[ feature_label ] # 스케일링(표준화) 한 데이터 독립변수
Y = data[ 'Species' ] # 종속변수
# 훈련용 데이터 , 평가용 데이터 분할
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split( X , Y , test_size=0.3 , random_state=0 )


# 결정트리분석
# 모델 객체 생성
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier( random_state=42 )
# 훈련용 데이터를 피팅(훈련) 하기
model.fit( x_train , y_train )
# 예측 하기 # 평가용으로 수행
y_predict = model.predict( x_test ) # 이진 분류 예측
# print( y_predict ) # [1. 1. 1. ... 1. 1. 1.]
print( f'개선 전 결정트리모델 정확도 : { accuracy_score( y_test , y_predict ) } ' ) # 예측 정확도 확인

# [7] 시각화
import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree( model  , feature_names= feature_label  , class_names= target_label )
plt.show()

# 개선
# 12 (모델의 성능 개선) 최적의 하이퍼 파라미터 찾기2
params = {
    'max_depth' : [  2 , 6 , 10 , 14  ],       # 트리의 최대 깊이 로 검증 하겠다.
    'min_samples_split' : [ 2 , 4 , 6 , 8  ]
}
from sklearn.model_selection import GridSearchCV
grid_cv = GridSearchCV( model , param_grid = params , scoring='accuracy' , cv = 5 , return_train_score= True )
grid_cv.fit( x_train , y_train )
print( f'평균 정확도:{ grid_cv.best_score_}, 최적 하이퍼 파라미터:{grid_cv.best_params_}' ) # 평균 정확도 : 0.8548794147162603 {'max_depth': 8, 'min_samples_split': 16}

# 예시] 개선된 모델 생성
model2 = DecisionTreeClassifier( max_depth=10 , min_samples_split = 4 , random_state= 42 )
model2.fit( x_train , y_train ) # 개선된 모델로 다시 피팅
    # 개선된 모델로 다시 테스트
Y_predict2 = model2.predict( x_test )           # 예측
print( f'개선 후 결정트리모델 정확도 : { accuracy_score( y_test , Y_predict2 ) } ' ) # 예측 정확도 확인

# ----------------------- * 결정트리 모델 시각화
import matplotlib.pyplot as plt
from sklearn import tree # 결정트리 시각화 모듈
tree.plot_tree( model2  , feature_names= feature_label  , class_names= target_label )
plt.show()
