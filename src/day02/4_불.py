# 4_불.py
# - 참True 과 거짓False 을 나타 내는 자료형
# True : 참을 의미 , False : 거짓을 의미 , *첫글자를 대문자로 시작 true[x] True[o]
'''
    - 자료형의 참과 거짓
    "python"    True        ""          False
    [1,2,3]     True        []          False
    (1,2,3)     True        ()          False
    {'a':1}     True        {}          False
    1           True        0           False
                            None        False
'''
# (1) 불 형태
a = True
b = False
# (2) type( 자료 ) , 자료의 타입 반환하는 함수
print( type( a ) )  # <class 'bool'>
print( type( b ) )  # <class 'bool'>
print( 1 == 1 ) # True
print( 2 > 1 )  # True
print( 2 < 1 )  # False
# (3) 자료형의 참과 거짓
# (4) 불 연산 , bool( 자료 ) : 해당 자료의 불타입의 T/F 반환 함수
print( bool('python') ) # True
print( bool('') )       # False
print( bool( [1,2,3] ) )    # True
print( bool( [] ) )         # False
print( bool( 0 ) )          # False
print( bool( 1 ) )          # True










