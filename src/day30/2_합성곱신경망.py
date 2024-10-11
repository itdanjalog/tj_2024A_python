# day30 --> 2_합성곱신경망

import tensorflow as tf
# 데이터셋 # 10가지의 종류의 이미지 데이터셋[비행기0,자동차1,새2,고양이3,사슴4,개5,개구리6개,말7,배8,트럭9]
cifar10 = tf.keras.datasets.cifar10
( x , y ) , (x_t , y_t) = cifar10.load_data() # (50000, 32, 32, 3)
# 칼라 이미지의 합성곱 모델 만들기
# 2. 정규화
x = x / 255.0
x_t = x_t / 255.0
# 3. 모델
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32 , ( 3,3) , activation = 'relu' , input_shape= ( 32,32,3 ) ) ,
    tf.keras.layers.MaxPooling2D( (2,2) ) ,
    tf.keras.layers.Dense( 128 , activation='relu'),
    tf.keras.layers.Dense( 64 , activation='relu' ) ,
    tf.keras.layers.Dense( 32 , activation='relu' ) , # accuracy: 0.7693 - loss: 0.6721 - val_accuracy: 0.6481 - val_loss: 1.0502
    tf.keras.layers.Flatten() ,
    tf.keras.layers.Dense( 10 , activation='softmax')
])
# 4. 컴파일
model.compile( optimizer = 'adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'] )
# 5. 피팅
model.fit( x , y , validation_data=( x_t , y_t ) , epochs = 10 )

############################## 외부 '개' 이미지의 예측
# 1. 파이썬 OpenCV : 이미지파일을 파이썬으로 호출 하는 모듈 제공 한다.
import cv2 # opencv-python 설치
# 2. 외부 이미지 가져오기
img = cv2.imread('dog.png')
print( img )
print( img.shape ) # (336, 575, 3) : 원본이미지는 가로336픽셀 세로575픽셀 칼라(3채널)
# 3. 이미지의 사이즈 변경
img = cv2.resize( img , dsize=( 32 , 32) ) # 모델이 학습한 사이즈와 동일하게 변경 # (32, 32, 3) 픽셀 줄이기
print( img.shape )
# * 정규화
img = img / 255.0
# 4. 변경된 이미지 cv시각화
# cv2.imshow( 'img' , img )
# cv2.waitKey()
# 5. 모델을 이용한 새로운 이미지 예측하기
result = model.predict( img[ tf.newaxis , ... ] ) # ( 32 , 32 , 3 ) --> ( 1 , 32 , 32 , 3 )
print( tf.argmax( result[0]).numpy() ) # 가장 높은 확률을 가진 종속변수
# 1. 정규화 안했더니 예측값:8 # 정규화 이후 에도 예측값 : 8
# 2. 레이어 추가했더니 예측값 5
############################## 외부 '자동차' 이미지의 예측
img = cv2.imread('car.png')
img = cv2.resize( img , dsize=( 32 , 32) )
img = img / 255.0
result = model.predict( img[ tf.newaxis , ... ] ) # ( 32 , 32 , 3 ) --> ( 1 , 32 , 32 , 3 )
print( tf.argmax( result[0]).numpy() ) # 가장 높은 확률을 가진 종속변수 # 1




















