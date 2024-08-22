# day07 > task7 > main.py
# 다른 파일의 클래스 가져오기
from user import User # User 클래스 가져오기

names = [ ]
def nameCreate( ) :
    global names
    name = input('newName : ')
    age = int( input('newAge : ') )
    user = User(name , age) # 객체 생성
    names.append(  user )  # 생성된 객체 리스트에 담기
    return
def nameRead( ) :
    global names
    for user in names :
        print( f'name : { user.name }  age : { user.age }')
    return
if __name__ == "__main__" :
    while True :
        ch = int( input('1.create 2.read') )
        if ch == 1 : nameCreate( )
        elif ch == 2 : nameRead( )