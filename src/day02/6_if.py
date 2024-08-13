# 6_if.py
'''
1. 자바
boolean money = true;
if( money ) {
    System.out.print("택시를 타고 가라");
}
else {
    System.out.print("걸어가라");
}

2. 파이썬
money = True # 변수
if money :
    print('택시를 타고 가라')
else :
    print('걸어가라')
'''
#
money = True  # 변수
if money:
    print('택시를 타고 가라')
else:
    print('걸어가라')
# money변수의 값이 True 이므로 '택시를 타고 가라' 출력된다.

# (1) if의 기본구조
'''
if 조건문 : 
(들여쓰기)수행문;
else : 
(들여쓰기)수행문;
'''
# (2) 들여쓰기 방법
    # 1. if문에 속하는 모든 실행문은 들여쓰기를 해야한다.
    # 2. 주의할점 : 다른 프로그래밍 언어를 사용해온 사람들은 무시 하므로 주의하자.
    # 3. tab(탭)키 , 파이참/인텔리제이 에서 코드 범위 지정후 ctrl+alt+L
    # 4. 들여쓰기 깊이/수준을 속해있는 범위 맞게 사용하자.
if money :
    print('택시를')
print('타고')
#    print('가라') # IndentationError: unexpected indent , 예외발생
if money :
    print('택시를')
    print('타고')
#        print('가라') # IndentationError: unexpected indent , 예외발생

# (3) 조건문이란 무엇인가?
    # - 조건문이란 참 과 거짓을 판단하는 문장
# 1. 비교연산자 , >초과 <미만 >=이상 <=이하 ==같다 !=같지않다
x = 3
y = 2
print( x > y )  # True
print( x < y )  # False
print( x == y ) # False
print( x != y ) # True
print( x >= y ) # True
print( x <= y ) # False
# 예제1
money = 2000
if money >= 3000 :      # money(2000) >= 3000  , False
    print('택시를 타고 가라')
else :
    print('걸어가라')
    # '걸어가라' 출력된다.

# 2. 논리연산자 , and 이면서 or 이거나 not 부정
money = 2000
card = True
if money >= 3000 or card : # money(2000) >= 3000 or card(True) , True
    print('택시를 타고 가라')
else :
    print('걸어가라')
# '택시를 타고 가라' 출력된다.

# 3. 기타연산자 , value in 리스트/튜플/문자열 , not in 리스트/튜플/문자열
print( 2 in [1, 2, 3] )     # True
print( 2 not in [1, 2, 3] ) # False
print( 'a' in ('a','b','c' ) ) # 튜플에 'a'가 있는가? True
print( 'j' not in 'python' )    # 문자열에 'j'가 없는가? True
# 예
pocket = ['paper' , 'cellphone' , 'money' ]
if 'money' in pocket :  # 만약에 리스트에 money 가 있으면 , True
    print('택시를 타고 가라')
    pass                # pass 이후의 실행문은 실행되지 않는다.
else :
    print('걸어가라')
# (4) 다양한 조건을 판단하는 elif
pocket = ['paper' , 'cellphone']
card = True
if 'money' in pocket :
    print('택시를 타고 가라`')
elif card :
    print('택시를 타고 가라')
else :
    print('걸어가라')

# (5) 조건부 표현식 , if 대신에 간단한 조건문 표현
    # 참일경우 if 조건문 else 거짓일경우
score = 80
message = 'success' if score >= 60 else 'failure'
print( message )    # success





















































