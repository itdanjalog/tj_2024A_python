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




