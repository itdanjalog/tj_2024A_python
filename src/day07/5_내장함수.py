# day07 > 4_내장함수.py
# 파이썬 배포본에 함께 들어있는 함수들 = 라이브러리
# import를 하지 않아도 된다.

# 1. abs( 숫자 ) : 절대값 함수
print( f'{ abs(3) } , { abs(-3) } , { abs(-1.2) }' )
# 2. all( 리스트/튜플/문자열/딕셔너리/집합 ) : 모두 참이면 참 반환 하는 함수
    # 데이터 들의 참과 거짓 , day02 4.불.py 참고
print( f'{ all( [1,2,3] ) } , { all( [1,2,3,0] ) } , {all([])}')
# 3. any( 리스트/튜플/문자열/딕셔너리/집합 ) : 하나라도 참이면 참 반환 하는 함수
print( f'{ any( [1,2,3] ) } , { any( [1,2,3,0] ) } , {any([])}')
# 4. chr(유니코드) : 유니코드 숫자 를 문자로 반환 하는 함수
print( f'{ chr(97) } , { chr(44032 ) }' ) # a , 가
# 5. dir( 객체 ) : 해당 객체가 가지는 변수 나 함수를 보여주는 함수
print( f'{ dir( [] ) } , { dir( {} ) }')
# 6. divmod( a , b ) : a를 b로 나눈 몫 과 나머지를 튜플로 반환
print( divmod( 7 , 3) ) # 몫:2 , 나머지:1   , ( 2,1 )
# 7. enumerate( 리스트/튜플/문자열 ) : 인덱스 값을 포함한 객체를 반환 한다.
for i , name in enumerate( ['body' , 'foo' , 'bar'] ) :
    print( i  , name )
# 8. eval( 문자열로 구성된 코드 ) :
print( eval( '1+2' ) )          # '1+2' -> 1+2 => 3
print( eval( "'hi'+'a'" ) )     # hia
print( eval( 'divmod(4,3)') )   # (1, 1)

# 9. filter( 함수 , 데이터 ) : 함수내 조건이 충족하면 데이터를 반환 함수  , list 타입 으로 변환 가능
def positive( x ) :
    return x > 0
data = [ 1, -3 , 2 , 0 , -5 , 6 ]
result = filter( positive , data )
print( list( result ) ) # list( ) : 리스트 타입으로 반환 해주는 함수
# 람다식 함수 , 함수명 = lambda 매개변수1,매개변수2 : 실행문
    # 주로 간단한 함수를 간결하게 사용한다.
add = lambda a , b : a + b  # return 명령어가 없어도 결과값이 리턴 된다.
print( add( 3,4) ) # 7
# filter 와 람다식 활용
result = filter( lambda x : x > 0 , data )   #  js : data.filter( x => x>0 )
print( list(result) )       # [1, 2, 6]

# 10. map( 함수 , 데이터 ) : 함수내 실행문 데이터를 반환 함수 , list 타입 으로 변환 가능
result = map( lambda x : x*2 , data )
print( list(result) )       # [2, -6, 4, 0, -10, 12]

# 11. hex

# 12. id

# 13. input

# 14. int

# 15. isinstance

# 16. len

# 17. list

# 18. max

# 19. min

# 20. oct

# 21 open

# 22. ord

# 23 pow

# 24. range

# 25. round

# 26. sorted

# 27. str

# 28. sum

# 29. tuple

# 30 type

# 31 zip















