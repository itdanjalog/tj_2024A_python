

#
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


import cv2 # opencv-python 설치
# 이미지 호출
img = cv2.imread('bag.jpg')
# 이미지의 사이즈 변경
img = cv2.resize( img , dsize=( 28 , 28) )
# 흑백으로 변환 (채널 수를 1로 변경)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img[..., tf.newaxis]
# 정규화
img = img / 255.0
img = img[ tf.newaxis , ... ]
result = model.predict( img )
# 예측
print( tf.argmax( result[0]).numpy() )