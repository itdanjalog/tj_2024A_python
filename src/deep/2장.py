
## ======================= 2-1
import tensorflow as tf
print(tf.executing_eagerly())

a = 1
b = 2
c = tf.math.add(a, b)
print(c)

c.numpy()

## ======================= 2-2
# 텐서플로 불러오기
import tensorflow as tf

# 스칼라 정의하기
a = tf.constant(1)
b = tf.constant(2)
print("a:", a)
print("b:", b)

# 랭크 확인하기
print(tf.rank(a))


# 자료형 변환
a = tf.cast(a, tf.float32)
b = tf.cast(b, tf.float32)
print(a.dtype)
print(b.dtype)

# 덧셈
c = tf.math.add(a, b)
print("result:", c)
print("rank:", tf.rank(c))

# 뺄셈
print( tf.math.subtract(a, b) )
# 곱셈
print( tf.math.multiply(a, b) )
# 나눗셈
print( tf.math.divide(a, b) )
# 나눗셈 (나머지)
print( tf.math.mod(a, b) )
# 나눗셈 (몫)
print( tf.math.floordiv(a, b) )

## ======================= 2-3
# 라이브러리 불러오기
import tensorflow as tf
import numpy as np

# 1차원 배열 정의
py_list = [10., 20., 30.] # 파이썬 리스트 활용
num_arr = np.array([10., 10., 10.]) # 넘파이 배열 활용

# 텐서 변환
vec1 = tf.constant(py_list, dtype=tf.float32)
vec2 = tf.constant(num_arr, dtype=tf.float32)

# 텐처 출력
print("vec1:", vec1)
print("vec2:", vec2)

# 랭크 확인
print(tf.rank(vec1))
print(tf.rank(vec2))

# 덧셈 함수
add1 = tf.math.add(vec1, vec2)
print("result:", add1)
print("rank:", tf.rank(add1))

# 덧셈 연산자
add2 = vec1 + vec2
print("result:", add2)
print("rank:", tf.rank(add2))

# tf.math 모듈 함수
print(tf.math.subtract(vec1, vec2))
print(tf.math.multiply(vec1, vec2))
print(tf.math.divide(vec1, vec2))
print(tf.math.mod(vec1, vec2))
print(tf.math.floordiv(vec1, vec2))

# 파이썬 연산자
print(vec1 - vec2)
print(vec1 * vec2)
print(vec1 / vec2)
print(vec1 % vec2)
print(vec1 // vec2)

# 합계 구하기
print ( tf.reduce_sum(vec1) )
print ( tf.reduce_sum(vec2) )

# 거듭제곱
print( tf.math.square(vec1) )

# 거듭제곱 (파이썬 연산자)
print( vec1**2 )

# 제곱근
print( tf.math.sqrt(vec2) )

# 제곱근 (파이썬 연산자)
print( vec2**0.5 )

# 브로드캐스팅 연산
print (vec1 + 1)

## ======================= 2-4
# 라이브러리 불러오기
import tensorflow as tf

# 2차원 배열 정의
list_of_list = [[10, 20], [30, 40]]

# 텐서 변환 - constant 함수에 2차원 배열 입력
mat1 = tf.constant(list_of_list)

# 랭크 확인
print("rank:", tf.rank(mat1))

# 텐서 출력
print("mat1:", mat1)

# 1차원 벡터 정의
vec1 = tf.constant([1, 0])
vec2 = tf.constant([-1, 2])

# 텐서 변환 - stack 함수로 1차원 배열을 위아래로 쌓기
mat2 = tf.stack([vec1, vec2])

# 랭크 확인
print("rank:", tf.rank(mat2))

# 텐서 출력하기
print("mat2:", mat2)

# element-by-element 연산
element_mul = tf.math.multiply(mat1, mat2)
print("result:", element_mul)
print("rank:", tf.rank(element_mul))

# 브로드캐스팅 연산
element_bc = tf.math.multiply(mat1, 3)
print("result:", element_bc)
print("rank:", tf.rank(element_bc))

# 행렬곱 연산
mat_mul = tf.matmul(mat1, mat2)
print("result:", mat_mul)
print("rank:", tf.rank(mat_mul))

# 덧셈 연산
add1 = tf.math.add(mat1, mat2)
print("result:", add1)
print("rank:", tf.rank(add1))

# 덧셈 연산자
add2 = mat1 + mat2
print("result:", add2)
print("rank:", tf.rank(add2))

# 텐서를 넘파이로 변환
np_arr = mat_mul.numpy()
print(type(np_arr))
print(np_arr)


## ======================= 2-5
# 라이브러리 불러오기
import tensorflow as tf
import numpy as np

# 2차원 배열 정의
mat1 = [[1, 2, 3, 4],
        [5, 6, 7, 8]]

mat2 = [[9, 10, 11, 12],
        [13, 14, 15, 16]]

mat3 = [[17, 18, 19, 20],
        [21, 22, 23, 24]]

# 텐서 변환 - constant 함수에 3차원 배열 입력
tensor1 = tf.constant([mat1, mat2, mat3])

# 랭크 확인
print("rank:", tf.rank(tensor1))

# 텐서 출력
print("tensor1:", tensor1)

# 텐서 변환 - stack 함수로 2차원 배열을 위아래로 쌓기
tensor2 = tf.stack([mat1, mat2, mat3])

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
tensor3 = tf.constant(arr)

# 랭크 확인
print("rank:", tf.rank(tensor3))

# 텐서 출력
print("tensor3:", tensor3)

# 랭크-4 텐서 만들기
tensor4 = tf.stack([tensor1, tensor2])

# 랭크 확인
print("rank:", tf.rank(tensor4))

# 텐서 출력
print("tensor4:", tensor4)

## ======================= 2-6
# 텐서플로 불러오기
import tensorflow as tf

# 벡터 정의하기
vec = tf.constant([10, 20, 30, 40, 50])
print(vec)

print(vec[0])

print(vec[-1])

print(vec[:3])

# 행렬 정의하기
mat = tf.constant([[10, 20, 30],
                    [40, 50, 60]])

print(mat[0, 2])

print(mat[0, :])
print(mat[:, 1])
print(mat[:, :])

# 랭크-3 텐서 정의하기
tensor = tf.constant([
    [[10, 20, 30],
     [40, 50, 60]],
    [[-10, -20, -30],
     [-40, -50, -60]],
])
print(tensor)

print(tensor[0, :, :])
print(tensor[:, :2, :2])

## ======================= 2-7
# 텐서플로 불러오기
import tensorflow as tf

# 랭크-1 텐서 정의하기
tensor = tf.constant(range(0, 24))
print(tensor)

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

## ======================= 2-8
# 텐서플로 불러오기
import tensorflow as tf

# 텐서 정의하기
tensor1 = tf.constant([[0, 1, 2],
                      [3, 4, 5]])
print(tensor1)

tensor_var1 = tf.Variable(tensor1)
print(tensor_var1)

print("이름: ", tensor_var1.name)
print("크기: ", tensor_var1.shape)
print("자료형: ", tensor_var1.dtype)
print("배열: ", tensor_var1.numpy())

tensor_var1.assign([[1, 1, 1],
                    [2, 2, 2]])
print(tensor_var1)

tensor2 = tf.convert_to_tensor(tensor_var1)
print(tensor2)

tensor_var2 = tf.Variable(tensor2, name='New Name')
print(tensor_var2.name)

print( tensor_var1 + tensor_var2 )

## ======================= 2-9
# 라이브러리 불러오기
import tensorflow as tf

# Y= 3X - 2 선형 관계를 갖는 데이터셋 생성
g = tf.random.Generator.from_seed(2020)
X = g.normal(shape=(10, ))
Y = 3 * X - 2

print('X: ', X.numpy())
print('Y: ', Y.numpy())


# Loss 함수 정의
def cal_mse(X, Y, a, b):
        Y_pred = a * X + b
        squared_error = (Y_pred - Y) ** 2
        mean_squared_error = tf.reduce_mean(squared_error)

        return mean_squared_error


# tf.GradientTape로 자동미분 과정을 기록

a = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 200

for epoch in range(1, EPOCHS + 1):

        with tf.GradientTape() as tape:
                mse = cal_mse(X, Y, a, b)

        grad = tape.gradient(mse, {'a': a, 'b': b})
        d_a, d_b = grad['a'], grad['b']

        a.assign_sub(d_a * 0.05)
        b.assign_sub(d_b * 0.05)

        if epoch % 20 == 0:
                print("EPOCH %d - MSE: %.4f --- a: %.2f --- b: %.2f" % (epoch, mse, a, b))
