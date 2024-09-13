# day21 -> 2_유방암진단분석.py
# # 11장. 회귀분석 (1) 로지스틱회귀분석
import pandas as pd
#[1] 데이터 준비하기
from sklearn.datasets import load_breast_cancer
b_cancer = load_breast_cancer() # 사이킷런 내장된 데이터 호출
#[2] 데이터 탐색하기
# print( b_cancer.DESCR ) # 내장 데이터의 설명서 호출
# 독립변수 : b_cancer.data 종속변수 :b_cancer.target
# print( b_cancer.data ) # 독립변수들 # 피처
# print( b_cancer.feature_names ) # 독립변수들의 이름
# print( b_cancer.target ) # 종속변수 # 타겟 # 진단결과
b_cancer_df = pd.DataFrame( b_cancer.data , columns=b_cancer.feature_names )
# print( b_cancer_df.head() )
b_cancer_df['diagnosis'] = b_cancer.target
# print( b_cancer_df.head() )
# print( b_cancer_df.shape ) # (569, 31) # (행개수, 열개수)
# print( b_cancer_df.info() )
#  - 전처리 과정 를 이용한 특정 값을 표준화한 후에 머신러닝 알고리즘( 로지스틱 회귀 , k-최근접 등등 ) 사용 하면 성능을 향상 할수 있다.
'''
    - 데이터의 평균 , 표준편차 , 원본값 
    - 평균 = 50 , 표준편차 10 , 원본값 60
    
                원본값-평균
    표준화값 =   ----------
                 표준편차 
                 
    - 데이터 크기가 크게 다른 경우 모델의 성능이 저하 될수 있으므로 사용된다.
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # 객체 생성
b_cancer_scaled = scaler.fit_transform( b_cancer.data )
# 데이터를 표준화 하기 # 각 특성 값을 평균이 0 분산이 1 이 되도록 변환 하는 과정 # 평균과 표준편차를 사용하여 표준화
# 표준화 전 vs 후
# print( b_cancer.data[0] )
'''
[1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
 1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
 6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
 1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
 4.601e-01 1.189e-01]
'''
# print( b_cancer_scaled[0] )
'''
[ 1.09706398 -2.07333501  1.26993369  0.9843749   1.56846633  3.28351467
  2.65287398  2.53247522  2.21751501  2.25574689  2.48973393 -0.56526506
  2.83303087  2.48757756 -0.21400165  1.31686157  0.72402616  0.66081994
  1.14875667  0.90708308  1.88668963 -1.35929347  2.30360062  2.00123749
  1.30768627  2.61666502  2.10952635  2.29607613  2.75062224  1.93701461]
'''
# [3] 로지스틱 회귀 분석 모델 구축
# 독립변수 와 종속변수
X = b_cancer_scaled # 스케일링(표준화) 한 데이터 독립변수
Y = b_cancer_df['diagnosis'] # 종속변수
# 훈련용 데이터 , 평가용 데이터 분할
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split( X , Y , test_size=0.3 , random_state=0 )
# 모델 객체 생성
from sklearn.linear_model import LogisticRegression
lr_b_cancer = LogisticRegression()
# 훈련용 데이터를 피팅(훈련) 하기
lr_b_cancer.fit( x_train , y_train )
# 예측 하기 # 평가용으로 수행
y_predict = lr_b_cancer.predict( x_test ) # 이진 분류 예측
# print( y_predict )
# [ 0 1 ~ ]
# print( lr_b_cancer.predict_proba( x_test )) # 이진 확률 예측
# [ [9.98656121e-01 1.34387864e-03] , [3.85539075e-02 9.61446092e-01] ~~~  ]

# [4] 모델의 성능 평가
# 1.
from sklearn.metrics import  confusion_matrix , accuracy_score , precision_score ,  recall_score , f1_score , roc_auc_score
print( confusion_matrix( y_test , y_predict ) ) #
'''
[[ 60   3]      [ [ TN , FP ]
 [  1 107]]       [ FN , TP ] ]
'''
# 2. 정확도
print( accuracy_score( y_test , y_predict ) )       # 0.9766081871345029    # 97% 이상    # 모델이 전체 데이터에서 얼마나 잘 예측 했는지??
# 3. 정밀도
print( precision_score( y_test , y_predict ) )      # 0.9727272727272728    # 97% 이상    # 모델이 양성으로 예측한 것들 중에서 실제 양성 비율
# 4. 재현율
print( recall_score( y_test , y_predict ) )         # 0.9907407407407407    # 97% 이상    # 실제 양성 중에서 모델이 얼마나 잘 양성으로 예측 했는지??
# 5. F1스코어
print( f1_score( y_test , y_predict ) )             # 0.981651376146789     # 97% 이상    # 정밀도 와 재현율의 균형
# 6. ROC 기반 AUC 스코어
print( roc_auc_score( y_test , y_predict ) )        # 0.9715608465608465    # 97% 이상    # 모델이 양성과 음성을 구별하는 능력 평가
    # 5 가지의 모델 평가 지수 계산 방법 # 1(100%)에 가까울수록 모델은 예측을 잘 표현하고 있다. # 대략 85% 이상이면 신뢰도 높다.
























