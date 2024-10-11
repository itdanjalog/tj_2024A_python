# day30 --> 1_합성곱신경망
import tensorflow as tf

# 1. 데이터셋 로드 , 10가지 종류의 의류 이미지 데이터셋
fashion_mnist = tf.keras.datasets.fashion_mnist
( x_train , y_train) , ( x_valid , y_valid) = fashion_mnist.load_data()
print( x_train.shape )  # 데이터의 차원 확인 # (60000, 28, 28) # 6만개의 가로28 픽셀 , 세로 28 픽셀 이미지
# 생각해보기 : 위 데이터셋을 이용한 합성곱 모델 구축하고 학습하여 정확도(accuracy) 95% 이상 되도록 최적의 하이퍼 파라미터 설정 하여 모델 만들기.
# 2. 데이터 전처리
x_train = x_train / 255.0 # 정규화  # 0~255 범위를 0 ~ 1 범위로 변경
x_valid = x_valid / 255.0
# 3. 모델
model = tf.keras.Sequential([  # Conv2D ( 특성맵수 , (커널영역단위) , activation='활성화함수' , input_shape=(입력텐서차원)
    tf.keras.layers.Conv2D( 32, (3,3) , activation = 'relu' , input_shape=( 28, 28 , 1 ) ), # 1. 합성곱 레이어 # input_shape=( 이미지가로픽셀 , 이미지세로픽셀 , 흑백1칼라3 )
    # 2. 풀링 레이어
    tf.keras.layers.MaxPooling2D( (2,2) ) , # 최대값 풀링 (2*2)
    # 3. 플래톤 레이어
    tf.keras.layers.Flatten(), # 학습된 결과의 다차원을 1차원으로 변환
    # 4. 출력 레이어
    tf.keras.layers.Dense( 10 , activation='softmax') # 종속변수의 결과 종류가 10개라서 10 , 다중분류 : sotfmax 활성화함수 사용한다.
])
# 4. 모델 컴파일 # 학습하면서 학습된 데이터를 가지고 테스트용 데이터를 실행 할수 있다.
model.compile( optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics=['accuracy'] )
# 5. 모델 훈련
history = model.fit(  x_train[ ... , tf.newaxis ] , y_train , # 학습에 사용되는 데이터
            validation_data = ( x_valid[ ... , tf.newaxis ] , y_valid  ) , # 학습하면서 컴파일이 테스트를 할 테스트 데이터
            epochs = 15 # 학습 반복수
           )
# 최적의 파라미터 찾기 위해서는 1. epochs 조정 2. 레이어 조정
# 1. epochs = n 개
# 2. tf.keras.layers.Dense( 32 , activation='relu' ) , # 32개 노드를 가지는 완전 연결 레이어 1개 추가
# 등등
# 적절한 정확도 떨어지지 않고 손실값이 증가하지 않는 지점 찾기.

# 정확도 와 손실 시각화
# 10. 손실과 정확도 시각화
import matplotlib.pyplot as plt
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
plot_loss_acc( history , 15 )











