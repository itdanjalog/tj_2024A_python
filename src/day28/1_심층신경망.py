# day28 -> 1_심층신경망.py # p.73

import tensorflow as tf

# 케라스의 내장된 데이터셋에서 mnist(손글씨 이미지) 데이터셋 로드
mnist = tf.keras.datasets.mnist
print( mnist )

# 데이터셋을 다운로드 해서 ( 훈련용 , 테스트용 )
( x_train , y_train),( x_test , y_test ) = mnist.load_data()
print( x_train.shape , y_train.shape )  # (60000, 28, 28) (60000,)
print( x_test.shape , y_test.shape )    # (10000, 28, 28) (10000,)
                                        # (데이터크기 , 세로픽셀 , 가로픽셀 )
                                        # 28*28 픽셀 크기의 정사각형 이미지 6만개 저장된 상태
# 시각 화
import matplotlib.pyplot as plt
fig  , axes = plt.subplots( 3 , 5 ) # 3행 5열 여러개 차트 표현
fig.set_size_inches( 8 , 5 ) # 전체 차트의 크기를 가로 8인치 세로 5인치

for i in range( 15 ) : # 0~14까지 반복문 실행
    ax = axes[ i//5 , i%5 ] # i//5 : 몫(행 인덱스 ) # i%5 : 나머지(열 인덱스 )
    # i=0 ,  0//5  ->  0  , 0%5 -> 0  [0,0]
    # i=1 ,  1//5  ->  0  , 1%5 -> 1  [0,1]
    # 1=2 ,  2//5  ->  0  , 2%5 -> 2  [0,2]
    ax.imshow( x_train[i] ) # ax.imshow() : 이미지를 차트에 출력하는 메소드
    ax.axis('off') # 축 표시 끄기
    ax.set_title( y_train[i] ) # 각 이미지(차트)/정답 를 제목으로 출력
plt.show()

# 데이터 전처리 # [0첫번째이미지 , 10:15특정한 픽셀 , : 전체 픽셀 ]
print( x_train[0 , : , : ] ) # 5 손글씨 출력

# 0 ~ 255 사이가 아닌 0 ~ 1 사이를 가질수 있도록 범위 를 정규화 하기
print( x_train.min() , x_train.max() ) #min() : 최소값 찾기 함수 # max() : 최대값 찾기 함수 # 0 255
# 데이터 정규화
x_train = x_train / x_train.max() #  값 / 최대값  # 각 값들의 나누기 255
print( x_train.min() , x_train.max() ) # 0.0 1.0
x_test = x_test / x_test.max() # 테스트용 정규화
print( x_train[ 0 , : , : ] ) # 5손글씨 정규화 후 출력

# Dense 레이어 에는 1차원 배열만 들어갈수 있으므로 2차원 배열을 1차원으로 변경
print( x_train.shape ) # (60000, 28, 28) # 2차원 ( 데이터수 , 가로 , 세로 )
# 방법1] 텐서플로 방법
print( x_train.reshape( 60000 , -1 ).shape ) # (60000, 784) # 1차원 ( 데이터수 , 가로 )
# 방법2 ] 플래톤 레이어 방법
print( tf.keras.layers.Flatten()(x_train).shape ) # (60000, 784)



















