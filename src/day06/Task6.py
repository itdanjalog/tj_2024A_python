
class User :
    pass

names = [ ]
def nameCreate( ) :
    global names
    return
def nameRead( ) :
    global names
    return
def nameUpdate(  ) :
    global names
    return
def nameDelete( ) :
    global names
    return
# 해당 파일을 다른 파일에서 호출했을때 호출 되지 않는 구역
    # 해당 파일을 직접 실행할때는 실행되는 구역
    # 해당 파일을 다른 파일에서 호출 할때 실행되지 않는 구역 [ 모듈 ]
if __name__ == "__main__" :
    while True :
        ch = int( input('1.create 2.read 3.update 4.delete : ') )
        if ch == 1 : nameCreate( )
        elif ch == 2 : nameRead( )
        elif ch == 3 : nameUpdate( )
        elif ch == 4 : nameDelete( )