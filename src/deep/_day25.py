# ======================================= 복습 =======================================
'''
- 텐서플로란
        1. 텐서플로(TensorFlow)는 구글이 개발한 오픈소스 딥러닝 프레임워크입니다.
        2. 주로 머신러닝과 딥러닝 모델을 만들고 훈련하는 데 사용됩니다.
        3. 텐서플로는 특히 신경망(neural network)을 기반으로 한 모델을 구축하고 학습시키는 데 강력한 도구로, 다양한 기계 학습 작업에 널리 사용됩니다.
        4. 텐서플로는 **케라스(Keras)**라는 고수준 API를 제공하여 보다 간단하게 모델을 정의하고 훈련할 수 있게 도와줍니다.
        5. 이 외에도 이미지 처리, 자연어 처리, 음성 인식 등의 다양한 분야에서 활용되고 있습니다.

- 텐서란
        1. **텐서(Tensor)**는 텐서플로(TensorFlow)에서 다루는 기본 데이터 구조입니다.
        2. 수학적으로는 다차원 배열이나 행렬로 생각할 수 있으며, 데이터를 여러 차원으로 표현할 수 있는 도구입니다.
        3. 텐서플로라는 이름 자체도 '텐서의 흐름'이라는 의미로, 데이터를 처리하고 학습시키는 과정을 나타냅니다.
        4. 종류
                1. 스칼라(0차원 텐서): 단일 숫자를 의미합니다. 예를 들어, 5는 스칼라 텐서입니다.
                2. 벡터(1차원 텐서): 숫자의 리스트로 생각할 수 있습니다. 예를 들어, [1, 2, 3]는 벡터입니다.
                3. 행렬(2차원 텐서): 2차원 배열로, 행(row)과 열(column)이 있는 구조입니다. 예를 들어, [[1, 2], [3, 4]]는 2차원 텐서입니다.
                4. 3차원 이상 텐서: 이미지나 비디오 데이터를 표현할 때 주로 사용되며, 예를 들어, 컬러 이미지는 세 개의 색상 채널(RGB)을 가지는 3차원 텐서로 표현됩니다.
                5. 텐서의 차원 수는 **랭크(rank)**라고 하며, 랭크가 높을수록 더 많은 차원을 가진 데이터를 표현할 수 있습니다.
                텐서플로에서는 이러한 텐서를 기반으로 수학적 연산을 수행하여 딥러닝 모델을 학습시키거나 데이터를 처리하게 됩니다.
'''
# 라이브러리 불러오기
import tensorflow as tf
# 1. 스칼라 # 0차원 텐서 # 상수 # 0랭크 # 0차수 # 방향 없음
a = tf.constant( 930 )
print( a )                      # tf.Tensor(930, shape=(), dtype=int32)
print( tf.rank(a) )             # tf.Tensor(0, shape=(), dtype=int32)
print( a.numpy()  )             # 930
print( tf.rank(a).numpy())      # 0

# 2. 벡터 # 1차원 텐서 # 1차원리스트[] # 1랭크 # 1차수 # 한 방향 # X 또는 Y # (행 또는 열)
a = tf.constant( [  0 , 9  , 3 , 0  ] )
print( a )                      # tf.Tensor([0 9 3 0], shape=(4,), dtype=int32)
print( tf.rank(a) )             # tf.Tensor(1, shape=(), dtype=int32)
print( a.numpy()  )             # [0 9 3 0]
print( tf.rank(a).numpy())      # 1

# 3. 행렬 # 2차원 텐서 # 2차원리스트[[]] # 2랭크 # 2차수 # 두 방향 # X 와 Y # (행,열)
a = tf.constant( [ [ 0 , 9  ] , [ 3 , 0 ] ] )
print( a )
'''
tf.Tensor(
[[0 9]
 [3 0]], shape=(2, 2), dtype=int32)
'''
print( tf.rank(a) )             # tf.Tensor(2, shape=(), dtype=int32)
print( a.numpy()  )
'''
[[0 9]
 [3 0]]
'''
print( tf.rank(a).numpy())      # 2

# ======================================= 2.5 고차원 텐서 =======================================
# 3. 고차원 텐서 # 3차원 텐서 # 3차원리스트[[[]]] # 3랭크 # 3차수 # 두 방향 # X 와 Y 와 Z # (높이,행,열)

# 2차원 배열 정의
mat1 = [[1, 2, 3, 4],[5, 6, 7, 8]]
mat2 = [[9, 10, 11, 12],  [13, 14, 15, 16]]
mat3 = [[17, 18, 19, 20],  [21, 22, 23, 24]]
# 텐서 변환 - constant 함수에 3차원 배열 입력
tensor1 = tf.constant( [mat1, mat2, mat3] ) # (3,2,4)
# 랭크 확인
print("rank:", tf.rank(tensor1))
# 텐서 출력
print("tensor1:", tensor1)
# 텐서 변환 - stack 함수로 2차원 배열을 위아래로 쌓기
tensor2 = tf.stack( [mat1, mat2, mat3] )
# 랭크 확인
print("rank:", tf.rank(tensor2))
# 텐서 출력
print("tensor2:", tensor2)

# 1차원 배열 정의
vec1 = [1, 2, 3, 4]
vec2 = [5, 6, 7, 8]
vec3 = [9, 10, 11, 12]
vec4 = [13, 14, 15, 16]
vec5 = [17, 18, 19, 20]
vec6 = [21, 22, 23, 24]
# 1차원 배열을 원소로 갖는 2차원 배열 정의
arr = [[vec1, vec2],
        [vec3, vec4],
        [vec5, vec6]]
# 텐서 변환
tensor3 = tf.constant(arr) # (3,2,4)
# 랭크 확인
print("rank:", tf.rank(tensor3))
# 텐서 출력
print("tensor3:", tensor3)

# 3. 고차원 텐서 # 4차원 텐서 # 4차원리스트[[[[]]]] # 3랭크 # 3차수 # 두 방향 # X 와 Y 와 Z 와 W # ( 축1 , 축2 , 축3 , 축4 )
# 랭크-4 텐서 만들기
tensor4 = tf.stack([tensor1, tensor2]) # (2,3,2,4)
# 랭크 확인
print("rank:", tf.rank(tensor4))
# 텐서 출력
print("tensor4:", tensor4)

# 3차원 텐서 생성 (예: 이미지 배치)
tensor_3d = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 4차원 텐서 생성 (예: 컬러 이미지 배치)
tensor_4d = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
# 텐서 차원 구분
print("3차원 텐서 차원 수:", tf.rank(tensor_3d).numpy())
print("4차원 텐서 차원 수:", tf.rank(tensor_4d).numpy())

# ======================================= 2.6 인덱싱 =======================================
# 벡터 정의하기
vec = tf.constant([10, 20, 30, 40, 50])
print(vec)
print(vec[0]) # 10
print(vec[-1]) # 50
print(vec[:3]) # 10 20 30

# 행렬 정의하기
mat = tf.constant([[10, 20, 30],[40, 50, 60]])
print(mat[0, 2]) # 30
print(mat[0, :]) # [10 20 30]  # (3,)
print(mat[:, 1]) # [20 50] # (2,)
print(mat[:, :]) # [[10, 20, 30],[40, 50, 60]] # (2,3)

# 랭크-3 텐서 정의하기
tensor = tf.constant([
    [[10, 20, 30],
     [40, 50, 60]],
    [[-10, -20, -30],
     [-40, -50, -60]],
])
print(tensor) # (2,2,3)
print(tensor[0, :, :]) # [[10, 20, 30],[40, 50, 60]] # (2,3)
print(tensor[:, :2, :2]) # [[[10, 20], [40, 50]], [[-10, -20], [-40, -50]]] # (2,2,2)

# ======================================= 2.7 형태변환 =======================================
# 랭크-1 텐서 정의하기
tensor = tf.constant(range(0, 24))
print(tensor)

# 개수가 안맞으면 오류 발생
#tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__Reshape_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input to reshape is a tensor with 24 values, but the requested shape has 21 [Op:Reshape]
tensor1 = tf.reshape(tensor, [3, 8])
print(tensor1)


tensor2 = tf.reshape(tensor1, [-1, 4])
print(tensor2)

tensor3 = tf.reshape(tensor2, [-1])
print(tensor3)

tensor4 = tf.reshape(tensor3, [-1, 3, 4])
print(tensor4)

tensor5 = tf.reshape(tensor4, [3, 2, 4])
print(tensor5)

tensor6 = tf.reshape(tensor5, [3, 2, 2, 2])
print(tensor6)

# ======================================= 2.8 변수 =======================================
# 텐서 정의하기
tensor1 = tf.constant([[0, 1, 2],
                      [3, 4, 5]])
print(tensor1)
'''
tf.Tensor(
[[0 1 2]
 [3 4 5]], shape=(2, 3), dtype=int32)
'''

tensor_var1 = tf.Variable(tensor1)
print(tensor_var1)
'''
<tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=
array([[0, 1, 2],
       [3, 4, 5]])>
'''

print("이름: ", tensor_var1.name) # 이름:  Variable:0
print("크기: ", tensor_var1.shape) # 크기:  (2, 3)
print("자료형: ", tensor_var1.dtype) # 자료형:  <dtype: 'int32'>
print("배열: ", tensor_var1.numpy()) # 배열:  [[0 1 2] [3 4 5]]

# 크기가 다르면 오류 : ValueError: Cannot assign value to variable ' Variable:0': Shape mismatch.The variable shape (2, 3), and the assigned value shape (2, 4) are incompatible.
tensor_var1.assign([[1, 1, 1],
                    [2, 2, 2]]) # 이 단어는 "할당하다" 또는 "배정하다"라는 뜻 # assign() 함수의 발음은 "어싸인"입니다.
print(tensor1) # 변수가 변경 되어도 텐서는 그대로 이다.
print(tensor_var1)

tensor2 = tf.convert_to_tensor(tensor_var1)
print(tensor2)

tensor_var2 = tf.Variable(tensor2, name='New Name')
print(tensor_var2.name)

print( tensor_var1 + tensor_var2 )

# ======================================= 2.9 자동 미분 =======================================

# Y= 3(기울기)X - 2(절편) 선형 관계를 갖는 데이터셋 생성
# 1. tf.random.Generator.from_seed(2020)로 랜덤 숫자 생성기를 만듭니다. 시드 값을 설정함으로써 항상 동일한 난수를 생성합니다.
g = tf.random.Generator.from_seed(2020)
# 2. X = g.normal(shape=(10, ))는 10개의 정규분포 난수를 생성하여 X에 저장합니다.
X = g.normal(shape=(10, ))
# 3. Y = 3 * X - 2는 X와 선형 관계를 갖는 Y 값을 생성하는데, 이 관계는 Y=3X−2입니다.
Y = 3 * X - 2
print('X: ', X.numpy())
# [-0.20943771  1.2746525   1.213214   -0.17576952  1.876984    0.16379918
#   1.082245    0.6199966  -0.44402212  1.3048344 ]
print('Y: ', Y.numpy())
# [-2.628313    1.8239574   1.6396422  -2.5273085   3.630952   -1.5086024
#   1.2467351  -0.14001012 -3.3320663   1.9145031 ]



# Loss 함수 정의
# **손실 함수(MSE, 평균 제곱 오차)**를 정의하는 함수입니다.
def cal_mse(X, Y, a, b):
        Y_pred = a * X + b # 계수(기울기) * X + 상수항(절편) # Y_pred = a * X + b: 주어진 a와 b를 이용해 예측값 Y_pred를 계산합니다.
        squared_error = (Y_pred - Y) ** 2 # 예측값 Y_pred와 실제 값 Y 간의 차이의 제곱(오차 제곱)을 계산합니다.
        mean_squared_error = tf.reduce_mean(squared_error) #  모든 오차 제곱의 평균을 계산하여 MSE를 반환합니다.

        return mean_squared_error


# tf.GradientTape로 자동미분 과정을 기록
# a와 b는 학습을 통해 최적화될 변수로, 처음에는 0으로 초기화합니다. a는 기울기(3에 수렴), b는 절편(-2에 수렴) 역할을 합니다.
a = tf.Variable(0.0) #변수
b = tf.Variable(0.0)

EPOCHS = 200

for epoch in range(1, EPOCHS + 1):

        # GradientTape: tf.GradientTape()는 자동 미분을 기록하는 역할을 합니다. 이 테이프를 사용해 손실 함수(MSE)를 기록합니다.
        with tf.GradientTape() as tape: # 손실 함수 계산 과정(MSE)을 기록하기 위해 테이프를 엽니다.
                mse = cal_mse(X, Y, a, b) # 현재 a와 b로 손실(MSE)을 계산합니다.

        # print( tape )

        # 기울기 계산
        # tape.gradient()를 이용하여 mse에 대한 a와 b의 미분값(기울기)을 구합니다.
        grad = tape.gradient(mse, {'a': a, 'b': b})   # 손실 함수의 a, b에 대한 기울기를 계산하여 딕셔너리로 반환합니다.
        print( grad )
        d_a, d_b = grad['a'], grad['b'] # 각각 a와 b의 기울기를 추출합니다.
        # print( d_a )
        # print( d_b )

        # 매개변수 업데이트: 경사하강법을 사용하여 a와 b를 업데이트합니다.
        # # a.assign(value) # a 변수의 값을 직접 value로 대체합니다. # 현재 변수의 값과 상관없이 새로운 값으로 덮어씁니다.
        # # a.assign_sub(value) # a 변수에서 value만큼을 감산합니다. # 현재 변수 a의 값을 기준으로 value를 빼고, 그 결과를 a에 다시 할당합니다.
        a.assign_sub(d_a * 0.05) # a 값을 d_a 기울기의 0.05배 만큼 감소시킵니다. 0.05는 학습률(learning rate)입니다.
        b.assign_sub(d_b * 0.05) # b 값도 마찬가지로 기울기의 0.05배 만큼 감소시킵니다.

        if epoch % 20 == 0:
                print("EPOCH %d - MSE: %.4f --- a: %.2f --- b: %.2f" % (epoch, mse, a, b))
# 이 코드는 랜덤하게 생성된 데이터셋
# X,Y에 대해 경사하강법을 사용하여 a와 b를 학습하고, 이를 통해
# Y=3X−2 관계를 찾아내는 선형 회귀 예시입니다. tf.GradientTape()를 이용해 자동 미분을 사용하여 기울기를 계산하고, 경사하강법을 통해 손실을 줄이며 학습합니다.

print( Y )
print( a * X + b )

import tensorflow as tf
import matplotlib.pyplot as plt

# 시각화
plt.scatter(X, Y, color='blue', label='R DATA')  # 실제 데이터 점
plt.plot(X, a * X + b, color='red', label='P DATA')  # 예측 직선
#plt.title("선형 회귀 결과")
plt.xlabel("X VALUE")
plt.ylabel("Y VALUE")
plt.legend()
plt.show()





# 실제 데이터 (X 값과 Y 값)
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]  # Y = 2X (간단한 선형 관계)


# Loss 함수 정의 (Mean Squared Error)
def cal_mse(X, Y, a, b):
    Y_pred = a * X + b
    mse = tf.reduce_mean((Y_pred - Y) ** 2)
    return mse


# 변수 초기화 (a, b)
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# 학습 반복 횟수 (Epochs)
EPOCHS = 200

# 경사하강법 학습 과정
for epoch in range(1, EPOCHS + 1):
    with tf.GradientTape() as tape:
        mse = cal_mse(X, Y, a, b)

    # 기울기 계산
    grad = tape.gradient(mse, [a, b])
    d_a, d_b = grad[0], grad[1]

    # 매개변수 업데이트 (학습률 0.01)
    a.assign_sub(d_a * 0.01)
    b.assign_sub(d_b * 0.01)

    # 20 Epoch마다 출력
    if epoch % 20 == 0:
        print(f"EPOCH {epoch} - MSE: {mse:.4f} - a: {a.numpy():.2f}, b: {b.numpy():.2f}")

# 최종 결과 출력
print(f"학습 완료 - a: {a.numpy():.2f}, b: {b.numpy():.2f}")

print( Y )
print( a * X + b )

# 시각화
plt.scatter(X, Y, color='blue', label='R DATA')  # 실제 데이터 점
plt.plot(X, a * X + b, color='red', label='P DATA')  # 예측 직선
#plt.title("선형 회귀 결과")
plt.xlabel("X VALUE")
plt.ylabel("Y VALUE")
plt.legend()
plt.show()

'''
    일차 함수 
            함수
    x  y
    1  1    x = 1 y = 
    2  2
    3  3
       4
       5
       
   y = 3x + 1
   x = 1 -> 4
   x = 2 -> 7
   = 기울기(기울어진 정도 )
    
    - 기울기
        y증가량
        ------
        x증가량
                            +3
    (-2,0)    (0,3)
                            +2
                            
    (-2,3)      (5,-2)      -5/7
    (-1,3)      (0,2)       +1/-1 
        - 오른쪽으로 내리면 음수
        - 오른쪽으로 올라가면 양수 
    
    - 일차 함수 식 구하기 
        - 기울기 m , 점(지나는점) a,b
        ( a , b ) ( x , y )
        - m = y증가량/x증가량
        - m = (y-b)/(x-a)
        - m * (x-a) = (y-b)/(x-a) * (x-a)
        - m(x-a) = (y-b)
        - m(x-a)+b = y
        - y = m(x-a)+b
            - 기울기 3 , ( 1 , 2 ) 
            - y = 3(x-1) + 2   # y = 기울기(x-x가지나는점) + y가지나는점
                - y = 3(x+2) -3
                - 3x+6-3
                - 3x+3
                - y = 3x+3
        - y = ax+b ( 단 a는 0이 아니다)                
                # x의 계수는 항상 기울기
                
        
        
    
        
        
    
        
   
   
       
'''



























