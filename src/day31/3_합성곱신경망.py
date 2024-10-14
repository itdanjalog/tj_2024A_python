# day31 --> 3_합성곱신경망.py

import tensorflow as tf
# 1. 데이터셋 로드 , 10가지 종류의 패션 이미지 데이터셋
# Label	Description
# 0	T-shirt/top : 티셔츠 # 1	Trouser : 바지 # 2	Pullover : 스웨터
# 3	Dress : 드레스 # 4	Coat : 코트 # 5	Sandal : 샌들
# 6	Shirt : 셔츠 # 7	Sneaker : 스니커즈 # 8	Bag : 가방 # 9	Ankle boot :부츠

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train , y_train ) , ( x_valid , y_valid) = fashion_mnist.load_data()

# 과제 : Functional Api 이용한 모델 생성( 다중 입력 ) 과 예측 테스트
# 모델을 생성 하고 훈련한 이후 인터넷 에서의 임의의 가방 사진 으로 예측 하여 예측 결과가 8이 나오도록 하시오.