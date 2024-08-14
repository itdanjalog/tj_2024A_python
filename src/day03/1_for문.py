# 1_for문.py
# - while 문과 비슷한 반복문
# - for문은 한눈에 들어온다는 장점

#(1) for문의 기본 구조
'''
for 변수 in 리스트(또는 튜플 또는 문자열 ) :
    실행문;
'''
# 예제1
test_list = ['one' , 'two' , 'three' ]
# 'test_list' 이라는 변수가  ['one' , 'two' , 'three' ] 를 참조한다.
for i in test_list :
    # : 콜론 다음의 실행문 작성시 들여쓰기 주의  , { } 없다.
    # 리스트내 문자열을 하나씩 문자열 타입으로 반환해서 반복처리 한다.
    print( i )
'''
(JS방식) 
let test_list = ['one' , 'two' , 'three' ]
test_list.forEach( i => { 
    console.log( i ); 
})
'''
# 예제2
a = [ (1,2) , (3,4) , (5,6) ]
# [요소1 , 요소2 , 요소3 ]    : 리스트 타입 ( 여러 요소를 저장 , 요소 수정/삭제 가능 )
# (요소1, 요소2, 요소3 )      : 튜플 타입 ( 여러 요소를 저장 , 요소 수정/삭제 불가능 ) - 고정형
for ( f , l ) in a :
    # 리스트내 튜플을 하나씩 ( 요소1 , 요소2 ) 튜플 타입으로 반환해서 반복처리 한다.
    # print( f , l )
    print( f + l )

# 예제3
marks = [ 90 , 25 ,67, 45 , 80 ] # 학생들의 점수 리스트
number = 0 # 학생 번호
for i in marks :
    # 들여쓰기 주의
    number += 1 # 학생 번호 1증가
    if i >= 60 : # 만약 에  i번째 요소의 값이 60점 이상이면
        print( f'{number}번 학생은 합격입니다.' )
    else :
        print( f'{number}번 학생은 불합격입니다.')
    # 파이썬의 if 조건문에 ( ) 가 없다 .

# (2) continue : 가장 가까운 for문의 처음으로 돌아가게 되는 키워드
number = 0
for mark in marks :
    number += 1
    if mark < 60 :
        continue
        # 가장 가까운 for문으로 이동 , continue 를 만나게 되면 아래 코드는 실행되지 않는다.
    print( f'{number}번 학생 축하합니다. 합격입니다.')

# (3) range( ) : 숫자 리스트를 생성하여 반환 해주는 함수
    # range( 숫자 ) : 0부터 숫자 미만 까지 포함하는 range 객체를 만들어준다.
    # range( 시작숫자 , 끝숫자 ) : 시작숫자 부터 끝 숫자 미만 까지 포함하는 range 객체를 만들어준다.
    # range( 시작숫자 , 끝숫자 , 증감단위 ) : 시작숫자 부터 끝 숫자 미만 까지 증감단위 만큼 증감하여 포함하는 range 객체를 만들어준다.
a = range( 10 )
print( a ) # 0,1,2,3,4,5,6,7,8,9
a = range( 1 , 11 )
print( a ) # 1,2,3,4,5,6,7,8,9,10

# 예제
for value in range( 10 ) :      # 0 ~ 9
    print( value , end=' ' ) # print(    , end =' ' ) : 줄바꿈 처리를 하지 않는 출력문
print( ) # 예제 구분
for value in range( 1 , 11 ) :  # 1 ~ 10
    print( value , end= ' ')
print( ) # 예제 구분
for value in range( 1 , 11 , 2 ) :  # 1 ~ 10 , 2씩증가 , 1 3 5 7 9
    print( value , end= ' ')
print( ) # 예제 구분
# 예제2 1 부터 10까지의 누적합계를 구하시오.
sum = 0
for i in range( 1, 11 ) :
    sum += i
print( sum )

# 예제3 1 부터 100까지의 누적합계를 구하시오.
sum = 0
for i in range( 1 , 101 ) : # 1 ~ 101미만
    # print( i )
    sum += i
print( sum )    # 5050

# 예제4 구구단
# 1. 단출력 : 2~9
for 단 in range( 2 , 10 ) :
    print( 단 )
# 2. 곱출력 : 1~9
for 곱 in range( 1 , 10 ) :
    print( 곱 )
# 3. for문의 중첩 : 단 마다 곱 인지? [O] 곱 마다 단 인지? [X]
for 단 in range( 2 , 10 ) :
    # 들여쓰기 주의
    for 곱 in range( 1 , 10 ) :
        # 들여쓰기 주의
        print( f'{단} X {곱} = { 단*곱:>2}' , end = '     ')
    print()

# (4) 리스트 컴프리헨션 사용 ,
    # [ 표현식 for 항목 in 반복가능객체 if 조건문 ]
    # [ 연산식 for 반복변수 in 리스트 if 조건문 ]
    # 2개 이상
        # [
        #   표현식 for 항목 in 반복가능객체 if 조건문
        #   for 항목 in 반복가능객체 if 조건문
        # ]
# 1.
a = [ 1 , 2 , 3 , 4 ]
result =  [ num*3 for num in a ]
# [] 안에서 for문을 사용한다.
# [ for 반복변수 in 리스트명 ]
print( result )
# 2.
result = [ i for i in a ]
# 반복되고 있는 i값을 하나씩 리스트 요소로 대입하여 리스트 생성 한다.
print( result )
# 3. 기존 리스트를 반복문을 활용하여 새로운 리스트 생성
print( [ i for i in a ] ) # JAVA/JS : 리스트명.map( ) 비슷하다.
# 4.
result = [ num for num in a if num%2 == 0 ]
print( result )
# 5. 2개 이상 , 구구단 예시
result = [ x * y for x in range( 2,10 )
                for y in range( 1,10 ) ]
print( result )















































