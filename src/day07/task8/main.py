# day07 > task8 > main.py
'''
    user.py : user 객체의 클래스 정의 하는 곳
    file.py : save() , load() 함수를 정의 하는 곳
    [조건1] 이름과 나이를 입력받아 names 에 저장하시오.
    [조건2] 프로그램이 종료되고 다시 실행해도 names 데이터가 유지되도록 파일처리 하시오.
'''
from user import User
from file import *
names = [ ]
def nameCreate( ) :
    newName = input( 'newName : ')
    newAge = int( input('newAge : '))
    user = User( newName , newAge )
    names.append( user )
    save( names ) # 파일처리 , 저장하기
    return
def nameRead( ) :
    for user in names :
        print( f'name : { user.name } age : { user.age } ')
    return

if __name__ == "__main__" :
    names = load() # 파일처리 , 불러오기
    while True :
        ch = int( input('1.create 2.read : ') )
        if ch == 1 : nameCreate( )
        elif ch == 2 : nameRead( )