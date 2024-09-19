# day22 -> 1_결정트리.py
'''
    1. 분류분석
        1. 로지스틱 회귀분석    : 주로 이진 분류
        2. 결정트리 분석       : 주로 다중 분류 , 여러개의 클래스로 분류
            - 피처 , 독립변수
            - 클래스 , 타겟 ,  , 종속변수
    2. 결정트리란?
        - 트리 구조 기반으로 의사 결정 해서 조건을 규칙노드로 나타내고 최종적인 리프노드 로 결과를 제공
            - 종류
            1. 루트 노드 : 트리의 최상단 위치 하는 노드 ,
            2. 내부/규칙 노드 : 속석(특징)에 기반해 데이터를 분할하는 기준 되는 노드
            3. 리프 노드 : 더 이상 분할되지 않고 최종적인 결과 노드
            - 노드 선택 기준 ( 2개중에 1개를 주로 사용 )
            1. 엔트로피 : 정보이득지수( 엔트로피가 줄어들어 얻게되는 이득) 가 높은 피처를 분할 기준으로 사용 #  DecisionTreeClassifier( criterion='entropy' ) 기본값
                -예] 사과및오렌지 믹스 쥬스 만드는데 섞인 주스의 맛이 얼마나 혼잡(혼란)스러운지 측정하는것이 엔트로피 입니다.
                    - 주스에 사과 와 오렌지가 섞여서 맛이 헷갈릴때 , 혼잡도가 높다 , 엔트로피가 높다.
                    - 주스에 사과 또는 오렌지 맛만 느껴질때 , 혼잡도이 낮다, 엔트로피가 낮다.
                - 즉] 불확실성을 측정하는 지표로, 값이 낮을수록 분류가 잘 된다는 것을 의미

            2. 지니 계수 : 지니계수가 낮은 피처를 분할 기준으로 사용 #  DecisionTreeClassifier( ) 기본값
                    - 주스에 사과 와 오렌지가 섞여서 맛이 헷갈릴때 , 예측이 어렵다. 지니 계수가 높다.
                    - 주스에 사과 또는 오렌지 맛만 느껴잘때 , 예측이 쉽다 . 지니 계수가 낮다.
                - 즉] 불순도를 측정하는 지표로 , 값이 낮을수록 분류가 잘 된다는 것을 의미.
'''
# [1] 데이터 샘플
data = {
    'size': [1, 2, 3, 3, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2],  # 과일의 크기
    'color': [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 2, 2, 3, 3],  # 1: 빨간색, 2: 주황색, 3: 노란색
    'labels': [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 2, 2]  # 0: 사과, 1: 오렌지, 2: 바나나
}
# [2] 데이터 프레임 생성
import pandas as pd
df = pd.DataFrame( data )
# [3] 독립변수/피처 , 종속변수/클래스/타겟 나누기
x = df[ ['size' , 'color'] ]
y = df[ 'labels' ]
# [4] 결정 트리 모델 생성 # DecisionTreeClassifier( c )
from sklearn.tree import DecisionTreeClassifier # 모델 모듈 호출
# DecisionTreeClassifier( criterion='entropy' ) # 엔트로피 방식
model = DecisionTreeClassifier( ) # 지니 계수 방식
# [8] 훈련용 , 테스트용 나누기
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size= 0.2 , random_state= 0 )
# [5] 모델 피팅
model.fit( x_train , y_train )
# [8] 예측
y_pred = model.predict( x_test )
# [9] 정확도
from sklearn.metrics import accuracy_score # 정확도 함수
accuracy = accuracy_score( y_test , y_pred ) # accuracy_score( 실제값 , 예측값 ) 정확도 확인
print( accuracy )  # 0.6666666666666666
# [6] 확인
print( model.get_depth() ) # 트리의 깊이 # 4
print( model.get_n_leaves() ) # 리프 노드의 개수 6
# [7] 시각화
import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree( model  , feature_names=['size' , 'color'] , class_names=['apple' , 'orange' , 'banana'])
plt.show()









