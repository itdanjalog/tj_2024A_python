#day31 --> 1_합성곱신명망.py # p.163
'''
- 딥러닝 프로세스(절차)
  1. 데이터 수집
  2. 데이터 전처리 : 수집된 데이터를 신경망 모델에 적합하게 수정
  3. 데이터 분할 : 훈련용 데이터 와 검증/데이트용 데이터 으로 나눈다. 주로 7:3 vs 8:2
  4. 모델 설계(구축)
    1. Sequential API( 이미 만들어진 클래스 ) , Functional API( 이미 만들어진 클래스 )
    2. 레이어 구성 : 입력 레이어  ---> 은닉 레이어 (conv2D,MaxPooling2D , Flatten )등등 ---> 은닉 레이어 ---> 출력 레이어  순으로 구성
    3. 활성화 함수 : 각 레이어 에서 학습된 값을 비선형으로 변환할때 사용. 주로 Relu , softmax 함수 사용
  5. 모델 컴파일 : 모델을 어떻게 학습 하고 평가 하는지 설정
    1. 옵티마이저 : 모델의 가중치를 업데이트하는 방법의 알고리즘/계산법 , adam :학습률 기반으로 최적화 알고리즘 , sgd : 확률적 경사 하강법
    2. 손실함수 : 실제값 과 예측과의 차이 , 분류모델 : sparse_categorical_crossentropy , 회귀 : mean_squared_error
    3. 평가지도 : 모델의 성능을 평가하는 지표 , 분류모델 : accuracy , 회귀 : mse
  6. 모델 학습
    1. 에포크 : 전체 훈련 데이터를 한 번 사용되는 것을 1 에포크 , 10에포크 이면 전체 훈련을 10번
    2. 검증 : validation_data 학습 중에 검증/테스트 데이터를 사용하여 모델의 손실,평가를 확인할수 있다.
  7. 모델 평가
    1. .evaluation( ) : 최종 성능의 손실함수 와 평가지표 결과를 볼수 있다.
-----> 모델 튜닝(하이퍼 파라미터)
    - 학습률 , 배치 크기 , 레이어 수 , 뉴런(노드) 수 , 활성화 함수 , 에포크 등등 여러 하이퍼파라미터 조정하기.
  8. 모델 예측
    1. .predict( )
'''
# 1. 데이터셋 준비
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
( x_train , y_train),(x_valid,y_valid) = mnist.load_data()
print( x_train.shape , y_train.shape )
print( x_valid.shape , y_valid.shape )

# 2. 새로운 출력값 배열을 생성 ( 홀수 :1 , 짝수 : 0 )
y_train_odd = [ ]
for y in y_train :
    if y % 2 == 0 :
        y_train_odd.append( 0 )
    else:
        y_train_odd.append( 1 )
y_train_odd = np.array( y_train_odd ) # 넘파이 배열

y_valid_odd = [ ]
for y in y_train :
    if y % 2 == 0 :
        y_valid_odd.append( 0 )
    else:
        y_valid_odd.append( 1 )
y_valid_odd = np.array( y_valid_odd ) # 넘파이 배열
# 3. 정규화
x_train = x_train / 255.0
x_valid = x_valid / 255.0
# 4. 채널 추가  # 마지막인덱스( -1 )의 새로운 축 추가
x_train_in = tf.expand_dims( x_train , -1 )
x_valid_in = tf.expand_dims( x_valid , -1 )
print( x_train_in.shape , x_valid_in.shape ) #  (60000, 28, 28, 1) (10000, 28, 28, 1)
















