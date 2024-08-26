import numpy as np
print( np.__version__ )

# 1.
ar1 = np.array( [1,2,3,4,5]); print( ar1 ); print( type(ar1) );
ar2 = np.array( [ [10,20,30] , [40,50,60] ] ); print( ar2 );
# 2.
ar3 = np.arange( 1 , 11 , 2 ); print( ar3 )
# 4.
ar4 = np.array( [1,2,3,4,5,6] ).reshape((3,2)); print( ar4 );
# 5.
ar5 = np.zeros((2,3)); print( ar5 )
# 1로 채워진 배열 생성
ones = np.ones((2, 3))
print("1로 채워진 배열:")
print(ones)

# 6.
ar6 = ar2[ 0:2 , 0:2 ]; print( ar6 )
# 7.
ar7 = ar2[ 0 , : ]; print( ar7 ) # ???
# 8.
ar8 = ar1 + 10; print( ar8 )
print( ar1 + ar8 )
print( ar8 - ar1 )
print( ar1 * 2 )
print( ar1 / 2 )
# 9.
ar9 = np.dot( ar2 , ar4 ); print( ar9 )


print( ar2.ndim ) # 차원
print( ar2.shape ) # 형태


print("평균:", np.mean(ar2))
print("표준편차:", np.std(ar2))
print("최대값:", np.max(ar2))
print("최소값:", np.min(ar2))


