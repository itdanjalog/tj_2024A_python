# 3_함수.py
# - 입력값을 가지고 어떤 일을 수행한 후 그 결과물을 내어 놓는 것
# - 사용목적 : 1.코드재활용(반복사용) 2.기능별분리(가독성 좋다)

# (1) 파이썬의 함수의 구조
'''
(1) py
def 함수명( 매개변수 , 매개변수 ) :
    실행문
    return 반환값 또는 생략
(2) JS
function 함수명( 매개변수 , 매개변수 ) {
    실행문
    return 반환값 또는 생략
}
(3) JAVA
반환타입 함수명( 타입 매개변수 , 타입 매개변수 ) {
    실행문;
    return 반환값 또는 생략
}
'''

# 예1 : 함수의 이름은 add 이고 입력으로 2개(a,b)의 값을 받으며 리턴값(a+b)은 2개의 입력값을 더한값이다.
# 함수 정의/만들기
def add( a , b ) :
    # { } 가 없으므로 들여쓰기 주의
    return a + b
# 함수 호출
add( 3 , 4 ) # add 함수에 3 과 4 를 매개변수로 전달하여 7 반환 받는다.
print( add( 3 , 4 ) ) # add 함수가 반환한 결과 값을 출력한다.
a = 3
b = 4
result = add( a , b )    # 리터럴 대신 변수를 대입해도 된다.
print( result ) # 7

# (2) 입력값과 리턴값의 따른 함수의 형태
# 1. 매개변수 O 리턴 O
def add( a , b ) :  # 함수 정의
    result = a + b
    return  result
# 함수 호출 시 인수2개 전달하여 반환값을 받아 'a'변수에 저장
a = add( 3 , 4 )
print( a )

# 2. 매개변수 X 리턴 O
# 함수 정의
def say( ) :
    return 'Hi'
# 함수 호출 시 인수 전달이 없고 반환값을 받아 'a'변수에 저장
a = say()
print( a )

# 3. 매개변수 O 리턴 X
# 함수 정의
def add( a , b ) :
    print( f'{ a } , { b } 의 합은 { a+b }입니다.')
# 함수 호출 시 인수2개 전달하여 반환값은 없다.
add( 3 , 4 )

# 4. 매개변수 X 리턴 X
# 함수 정의
def say( ) :
    print('Hi')
# 함수 호출 시 인수 없고 반환값도 없다.
say( )

# (3) 매개변수를 지정하여 호출 하기
def sub( a , b ):
    # print( a , b )
    return  a - b

# 함수 호출시 인수의 값를 대입할 매개변수를 지정
result = sub( b = 3 , a = 7 )
print( result )
result = sub( a = 7 , b = 3 )
print( result )

# (4) 입력값이 몇 개가 될지 모를 때 , 가변 매개변수

# 1. *매개변수 : 매개변수 이름 앞에 *을 붙이면 입력값을 전부 모아 튜플로 만들어준다.
def add_many( *args ) : # 함수 정의 , *매개변수
    print( args ) # 여러개의 매개변수 값이 들어있는 !튜플 타입
    result = 0 # 더한 값을 저장하기 위한 변수
    for i in args : # 여러개 매개변수 튜플 를 반복처리
        result += i # 누적합계
    # for 종료후
    return result # 함수 종료시 반환되는 값
# 함수 호출
result = add_many( 1,2,3 ); print( result )
result = add_many( 1, 2, 3, 4,5,6,7,8,9,10 ); print( result )

# 2. 특정 매개변수를 먼저 작성하고 뒤로 *여러개의매개변수 작성한다.
def add_mul( choice , *args ) : # 여러개의 매개변수와 특정 매개변수가 존재할때는
    print( choice ) # 1개의 자료 매개변수
    print( args )   # 여러개의 자료를 가지는 튜플(매개변수)
    if choice == 'add' :            # 만약에 매개변수 값이 add 이면
        result = 0
        for i in args :
            result += i             # 누적합계
    elif choice == 'mul' :          # 아니고 만약에 매개변수 값이 mul 이면
        # else if [x] elif [o]
        result = 1
        for i in args :
            result *= i             # 누적곱
    return result                   # 반환 값
# 함수 호출
result = add_mul( 'add' , 1 ,2 ,3 ,4, 5 ) # choice = 'add' , args = ( 1 ,2 ,3 ,4, 5 )
print( result )
result = add_mul( 'mul' , 1,2,3,4,5 )    # choice = 'mul' , args = ( 1 ,2 ,3 ,4, 5 )
print( result )

# (5) 키워드 매개변수 , kwargs키워드
# 1.  ** 인수로 전달된 키와값을 딕셔너리의 매개변수로 받는다.
def print_kwargs( **kwargs ) :
    print( kwargs )     # {'a': 1} 딕셔너리 타입으로 매개변수를 받는다.
                        # {'name': 'foo', 'age': 3}
print_kwargs( a = 1 )   # 인수로 전달시 키와 값으로 전달
print_kwargs( name='foo' , age = 3 )

# (6) 함수의 리턴값은 언제나 하나이다. 여러개 일때는 [],(),{} 활용
# 1.
def add_and_nul( a , b ) :
    튜플 = 1 , 2                 # ( ) 생략시 튜플로 생성된다.
    print( 튜플 )                # (1, 2)
    return a+b , a*b
# 함수호출
result = add_and_nul( 3 , 4 )
print(result)                   # (7, 12) , 튜플1개(튜플안에 요소2개)
# 2. 동일한 수준의 return는 하나만 존재 해도 된다.
def add_and_nul( a , b ) :
    return a + b
    return a * b # 위에 리턴이 존재하므로 해당 리턴은 실행되지 않는다.
# 3. 서로 다른 수준의 return은 여러게 존재 할 수 있다.
def add_and_nul( a , b ) :
    if a < 0 :
        return  # 만약에 a가 0보다 작으면 함수 강제 종료 , 아래 코드는 실행되지 않느다.
    return a + b

# (7) 매개변수에 초깃값 미리 설정하기 , 함수 정의할때 매개변수의 디폴트값 대입하기
    # 주의할점 : 초기화 하고 싶은 매개변수는 항상 뒤쪽에 놓아야한다.
        #  say_myself( name , age , man = True )  [ O ]
        #  say_myself( name , man = True , age )  [ X ] , 인자값과 매개변수 식별이 불가능
def say_myself( name , age , man = True ) :  # 함수 정의
    print( f'나의 이름은 {name}입니다.')
    print( f'나이는 {age}살입니다.')
    if man :
        print('남자입니다.')
    else:
        print('여자입니다.')
# 함수 호출
# 만약에 해당 매개변수의 인자값이 없으면 디폴트(초깃) 값이 대입된다.
say_myself( '박응용' , 27 ) # 나의 이름은 박응용입니다. 나이는 27살입니다. 남자입니다.
say_myself( '박응용' , 27 , False ) # 나의 이름은 박응용입니다. 나이는 27살입니다. 여자입니다.

# (8) 함수 안에서 선언한 변수의 효력 범위 , 지역변수
    # 함수 안에서 사용하는 매개변수는 함수 밖의 변수 이름과는 전혀 상관이 없다.
a = 1           # 전역변수 'a'
def vartest( a ) :
    a = a + 1   # 지역변수 'a'; a = 2 지역변수의 2는 함수 종료시 사라진다.
vartest( a )
print( a )      # 전역변수 'a'의 값 확인 , 1

# 함수안 에서 함수 밖 변수를 활용하는 방법 , 1번 방법 권장
#1. return ,
a = 1
def vartest( a ) :
    a = a + 1
    return  a   # 지역변수는 함수 종료시 사라지므로 함수를 호출했던 곳으로 반환
a = vartest( a )
print( a )      #  2

#2-1. global 키워드 : 함수 밖 변수를 함수 안으로 호출할때 사용하는 키워드
a = 1
def vartest( ) :
    global a    # 함수 밖에 있는 a변수를 함수 안에서 호출
    a = a + 1
vartest()
print( a )      # 2

#2-2. 지역변수의 특성 : 함수 밖에서 함수 안으로 접근 가능하지만, 함수 안에서 함수 밖으로는 접근이 불가능
b = 1
def vartest( ) :
    c = b + 1   # 함수 밖에 있는 b변수를 함수 안에서 호출 , global 없이도 가능.
    return c
b = vartest()
print( b )      # 2





















