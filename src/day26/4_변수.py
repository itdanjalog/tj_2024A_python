# day26 > 4_변수.py # p43
import  tensorflow as tf

# 1. 행렬 텐서
tensor1 = tf.constant( [ [0,1,2],[3,4,5] ] )
print( tensor1 ) # tf.Tensor( [[0 1 2][3 4 5]], shape=(2, 3), dtype=int32)

# 2. 텐서플로 변수 생성  #  tf.Variable( )
tensor_var1 = tf.Variable( tensor1 )
print( tensor_var1 ) # <tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=array([[0, 1, 2],[3, 4, 5]])>
# constant( ) 값 변경 불가능 , Variable ( ) 값 변경 가능

# 3. 텐서플로 변수 속성 확인
print( tensor_var1.name )       # 텐서플로 변수명 #Variable:0
print( tensor_var1.shape )      # 크기 # (2, 3)
print( tensor_var1.dtype )      # 데이터 자료형/타입 # <dtype: 'int32'>
print( tensor_var1.numpy() )    # 데이터 # [[0 1 2][3 4 5]]

# 4. 텐서플로 변수 데이터 변경/수정/새로운 할당 #  자료형 과 크기 동일 해야 한다.
tensor_var1.assign( [ [1,1,1],[2,2,2] ] )
print( tensor_var1 ) # <tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=array([[1, 1, 1],[2, 2, 2]])>
print( tensor1 ) # 그대로 # # tf.Tensor( [[0 1 2][3 4 5]], shape=(2, 3), dtype=int32)

# 5. 텐서플로 변수 ---> 텐서  변환 # tf.convert_to_tensor( 텐서플로변수 )
tensor2 = tf.convert_to_tensor( tensor_var1 )
print( tensor_var1 )# <tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=array([[1, 1, 1],[2, 2, 2]])>
print( tensor2 ) # tf.Tensor([[1 1 1][2 2 2]], shape=(2, 3), dtype=int32) # 텐서의 크기와 저장하고 있는 값 은 변경 불가능.

# 6. 텐서플로 변수에 name 속성 정의
tensor_var2 = tf.Variable( tensor2 , name='new Name' )
print( tensor_var2 ) # <tf.Variable 'new Name:0' shape=(2, 3) dtype=int32, numpy=array([[1, 1, 1],[2, 2, 2]])>

# 7. 텐서플로 변수의 연산
print( tensor_var1 + tensor_var2 ) # tf.Tensor( [[2 2 2] [4 4 4]], shape=(2, 3), dtype=int32)
    # [ [1,1,1],[2,2,2] ] + [ [1,1,1],[2,2,2] ]  =  [[2 2 2] [4 4 4]]
