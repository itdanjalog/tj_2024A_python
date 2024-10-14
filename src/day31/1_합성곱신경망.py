#day31 --> 1_합성곱신경망.py # p.163
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
for y in y_valid :
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

# 5. 모델 생성
################ 합성곱 입력 구조 #####################
# (1) 입력 레이어  # 객체
inputs = tf.keras.layers.Input( shape = (28 , 28 , 1) )
# (2) 합성곱 레이어
conv = tf.keras.layers.Conv2D( 32 , (3 , 3) , activation = 'relu')
conv = conv( inputs ) # 합성곱 레이어 앞에 입력레이어 연결 하기 # __call__ # 입력레이어 <-- 합성곱 레이어
# (3) 풀링 레이어
pool = tf.keras.layers.MaxPooling2D( (2,2) )
pool = pool( conv ) # 풀링 레이어 앞에 합성곱레이어 연결 하기 # __call__ # 입력레이어 <-- 합성곱 레이어 <-- 풀링 레이어
# (4) 플래톤 레이어
flat = tf.keras.layers.Flatten()
flat = flat( pool ) # 플래톤 레이어 앞에 풀링 레이어 연결 하기  # __call__ # 입력레이어 <-- 합성곱 레이어 <-- 풀링 레이어 <-- 플래톤 레이어
################ 단순 입력 구조 추가 #####################
flat_inputs = tf.keras.layers.Flatten()  # 입력레이어 <-- 플래톤 레이어
flat_inputs = flat_inputs( inputs )
################ 2개 입력 구조를 1개 출력으로 만들기 # 합치기  #####################
concat = tf.keras.layers.Concatenate()
concat = concat( [ flat , flat_inputs ] ) # 2개 입력 구조를 합치기
# 입력레이어 --> 합성곱 레이어 --> 풀링 레이어 --> 플래톤 레이어
#                                                          -----> Concatenate()입력레이어합치기 -----> Dense()출력 레이어
# 입력레이어 -------------------------------> 플래톤 레이어
# (5) 출력 레이어
outputs = tf.keras.layers.Dense( 10 , activation='softmax' )
outputs = outputs( concat )
# (6) 모델
model = tf.keras.models.Model( inputs = inputs , outputs = outputs)

# 6. 모델 컴파일
model.compile( optimizer = 'adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
# 7. 모델 훈련
history =  model.fit( x_train_in , y_train ,
                      validation_data = ( x_valid_in , y_valid ) ,
                      epochs = 10 )
# 8. 모델 성능
loss , acc = model.evaluate( x_valid_in , y_valid )
print( loss , acc )

# 다중 출력 분류 모델 ( 1. 다중분류[0~9] 2.이진분류[0,1] )

# 1. 모델 생성 # name속성은 모델객체내 레이어 호출시 사용한다.(레이어식별용)
    # (1) 입력 1
inputs = tf.keras.layers.Input( shape = (28,28,1) , name='inputs')
conv = tf.keras.layers.Conv2D( 32 , (3,3) , activation='relu' , name='conv2d_layer' )(inputs)
pool = tf.keras.layers.MaxPooling2D( (2,2) , name='maxpool_layer')(conv)
flat = tf.keras.layers.Flatten( name='flatten_layer')(pool)
    # (2) 입력 2
flat_inputs = tf.keras.layers.Flatten()(inputs)
    # (3) 합치기
concat = tf.keras.layers.Concatenate()( [flat , flat_inputs] )
    # (4) 출력 레이어 2개
digit_outputs = tf.keras.layers.Dense( 10 , activation='softmax' , name='digit_dense')(concat)
odd_outputs = tf.keras.layers.Dense( 1 , activation='sigmoid' , name='odd_dense')(flat_inputs)
    # (5) 모델 생성
model = tf.keras.models.Model( inputs = inputs , outputs = [ digit_outputs , odd_outputs ] )
# 확인

print( model.input )
print( model.output )


# 2. 모델 컴파일 # 다중 출력의 손실함수는 loss = { }
model.compile( optimizer = 'adam' ,
               loss = { 'digit_dense' : 'sparse_categorical_crossentropy' ,  'odd_dense' : 'binary_crossentropy' },
                loss_weights = { 'digit_dense' : 1 , 'odd_dense' : 0.5 } ,  # 손실함수 가중치 # 1 :100% , 0.5 : 50%
               # 0~9 예측/결과 는 100% 반영하고 홀짝예측/결과 는 50% 반영 설정 # 모델 손실계산에 사용할 비중(가중치)
                metrics = [ 'accuracy', 'accuracy'  ] ) # 다중 출력에 따른 평가지표를 다중 설정

print( model.summary() )

# 3. 모델 훈련 # 다중 출력시 훈련용과 검증용이 다중이 되므로 { '출력레이어name' : 출력레이어변수 } 딕셔너리 구조 사용.
history = model.fit( { 'inputs' : x_train_in } , {'digit_dense' : y_train , 'odd_dense' : y_train_odd } , # 훈련용
                     validation_data = ( { 'inputs' : x_valid_in } , {'digit_dense' : y_valid , 'odd_dense' : y_valid_odd }  ) ,
                     epochs = 10 )
# 4. 모델 성능 평가
model.evaluate(  { 'inputs' : x_valid_in } , {'digit_dense' : y_valid , 'odd_dense' : y_valid_odd }  )

# 5. 모델 예측
print( y_valid[ 0 ] ) # 정답 : 7
digit_preds , odd_preds = model.predict( x_valid_in ) # 예측
print( digit_preds[0] )
print( odd_preds[0] )














