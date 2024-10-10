# day29 --> 1_합성곱신경망.py
# 1. p.142 ~ p.149 개념 정리
# 2. p.150 ~ p.162  (p.157  model.input -> inputs , model.output -> outputs )

# 1. 텐서플로 모듈 호출
import tensorflow as tf
# 2. 데이터셋 # 손글씨
mnist = tf.keras.datasets.mnist
( x_train , y_train),(x_valid,y_valid) = mnist.load_data()
# x_train : 훈련용 28*28 픽셀된 0~9 숫자
# y_train : 훈련용 0-9 숫자
# x_valid : 테스트용 28*28 픽셀된 0~9 숫자 , 독립변수
# y_valid : 테스트용 0-9 숫자 , 종속변수
print( x_train.shape , y_train.shape ) # (60000, 28, 28) # 6만개의 28*28 이미지 - 3차원 (60000,) # 1차원
print( x_valid.shape , y_valid.shape ) # (10000, 28, 28) # 1만개의 28*28 이미지 - 3차원 (10000,) # 1차원
# 확인
print( x_train[ 0 , : , : ] ) # 첫번째 데이터의 손글씨 0~255 # 2차원형식으로 표현
print( y_train[ 0 ] )   # 첫번째 데이터의 손글씨 정답   # 5
# 3. 시각화
import matplotlib.pyplot as plt
def plot_image( data , idx ) :
    plt.imshow( data[idx] ) # 차트에 이미지 표현 함수 # imshow( ) # 5
    plt.show() # 차트열기
plot_image( x_train , 0 )

# 4. 정규화 # 각 픽셀 값을 255 으로 나누어 0 ~ 1 사이의 값으로 변환하여 모델 학습을 더 빠르고 안정적으로 만들기
print( x_train.min() , x_train.max() , x_valid.min() , x_valid.max() ) # 정규화 전 # 0 255 0 255
x_train = x_train / 255.0
x_valid = x_valid / 255.0
# y(종속변수)는 정규화 를 하지 않는다.( 정답이니까 )
print( x_train.min() , x_train.max() , x_valid.min() , x_valid.max() ) # 정규화 후 # 0.0 1.0 0.0 1.0

# 5. 채널 축 # 채널이란? 색상 정보를 가지는 구성 요소
# 흑백(모노컬러) 이미지를 위해 1개 채널 추가
print( x_train.shape , x_valid.shape )          # (60000, 28, 28) (10000, 28, 28)
x_train_in = x_train[ ... , tf.newaxis ] # 파이썬에서 배열에 축(차원) 추가하는 방법 # ... : 기존배열 데이터 뜻한다.
x_valid_in = x_valid[ ... , tf.newaxis ] # 3차원 ---> 4차원
print( x_train_in.shape , x_valid_in.shape )    # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 6. 모델 구축
# model = tf.keras.Sequential( [ 입력레이어 , 은닉레이어 , 은닉레이어  , 출력레이어 ] )
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32 , ( 3 , 3 ) , activation = 'relu' , input_shape = ( 28,28,1) , name='conv' ) , # 합성곱(Conv2D) 입력레이어
        # 32 , ( 3 , 3 ) : 32개의 필터를 가진 3*3 크기의 합성곱 레이어 추가
        # relu : Relu 활성화 함수를 사용
        # input_shape = ( 28,28,1 ) : 독립변수의 차원 모양 ( 3차원 ) , ( 가로 , 세로 , 채널 )
    # 풀링 레이어
    tf.keras.layers.MaxPooling2D( (2,2) , name='pool' ) ,
        # 2*2 크기의 최대 폴링 레이어 추가 , 특성맵 크기를 줄인다.
    # 플래톤 레이어
    tf.keras.layers.Flatten() , # 다차원 배열을 1차원 배열로 변환한다.
    # 출력 레이어
    tf.keras.layers.Dense( 10 , activation='softmax' )
        # 종속변수가 분류할 데이터가 0~9 이므로 10개  # 다중분류에서는 주로 softmax 활성화 함수를 사용한다.
])
# 7. 모델 컴파일 # 옵티마이저 , 손실함수 , 평가지표 설정
model.compile( optimizer = 'adam' , loss='sparse_categorical_crossentropy' , metrics = ['accuracy'])
    # 옵티마이저 : adam 옵티마이저 로 설정
    # 손실함수 : 분류모델의 오차 계산법인 엔트로피 설정
    # 평가지표 : 분류모델의 정확도 계산법인 accuracy 설정
# 8. 모델 훈련
history = model.fit( x_train_in , y_train , # 훈련용 데이터 와 훈련용 정답
           validation_data = ( x_valid_in  , y_valid ) ,  # 테스트용 데이터와 테스트용 정답
           epochs = 10  ) # 전체 데이터셋을 10회 반복하여 훈련한다.
# 9. 훈련된 손실 , 정확도 확인하기
print( model.evaluate( x_valid_in , y_valid ) ) # 테스트용 독립변수 와 테스트용 종속변수(정답) 를 평가하기. # [0.057617511600255966, 0.9829999804496765]

# 10. 손실과 정확도 시각화
def plot_loss_acc( history , epoch ) :
    loss = history.history['loss'] # 훈련 손실(오차) 값
    val_loss = history.history['val_loss'] # 테스트 손실(오차) 값
    acc = history.history['accuracy'] # 훈련 정확도
    val_acc = history.history['val_accuracy'] # 테스트 정확도
    # 서브플롯 차트 구성
    fig , axes = plt.subplots( 1 , 2 ) # 1행 2열로 구성된 서브플롯
    axes[0].plot( range(1,epoch +1 ) , loss  )  # x축 훈련수 # y축은 훈련 오차 값
    axes[0].plot( range(1,epoch +1 ) , val_loss )  # x축 훈련수 # y축은 테스트 오차 값

    axes[1].plot( range(1,epoch +1 ) , acc  )  # x축 훈련수 # y축은  정확도
    axes[1].plot( range(1,epoch +1 ) , val_acc )  # x축 훈련수 # y축은 테스트 정확도

    plt.show()

plot_loss_acc( history , 10 )

# 11. 훈련된 모델로 예측 하기
print( y_valid[0] )# 종속변수  #  10000개 중에 첫번째 손글씨의 정답 # 숫자
# 7
print( tf.argmax( model.predict( x_valid_in )[0] ) ) # 독립변수 # 테스트용으로 예측하기. # 이미지된 손글씨
# argmax( ) : 배열내 가장 큰 값을 가진 요소의 인덱스 반환
# tf.Tensor(7, shape=(), dtype=int64)















































