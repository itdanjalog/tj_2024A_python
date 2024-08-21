# day06 > Task3.py
# 튜플 활용 , p.89 ~ p.92
# [조건1] : 각 함수 들을 구현 해서 프로그램 완성하기
# [조건2] :  1. 하나의 이름을 입력받아 names 에 저장합니다.
#           2. 저장된 여러명의 이름들 names 을 모두 출력 합니다.
#           3. 수정할 이름을 입력받아 존재하면 새로운 이름을 입력받고 수정 합니다.
#           4. 삭제할 이름을 입력받아 존재하면 삭제 합니다.
# [조건3] : names 변수 외 추가적인 전역 변수 생성 불가능합니다.
# 제출 : git에 commit 후 카톡방에 해당 과제가 있는 링크 제출

# 튜플이란 ? 리스트와 비슷 , 차이점 : 요소의 삽입/삭제가 불가능
    # 튜플은 불변성을 가진다. 리터럴과 같다.
    # a = 3 + 2 ,               , 5
    # a = 'py' + '3'            , py3
    # a = ( 1 , 2 ) + ( 3 )     , ( 1,2,3)
names = ( '유재석' , '강호동' , '신동엽') # 샘플 데이터
def nameCreate( ) :
    global names
    newName = input('newName : ')
    names += ( newName , )  # 기존 튜플 과 입력받은값의 튜플 더하기 해서 새로운 튜플 반환
    return

def nameRead( ) :
    global names
    for name in names : # for 반복변수 in 튜플/리스트/문자열 :
        print( f'name : {name}' )
    return




def nameUpdate(  ) :
    global names
    oldName = input( 'oldName : ')
    if names.count( oldName ) == 0 : return
    else :
        newName = input( 'newName : ')
        newNames = ( )
        for name in names :
            if name == oldName :
                newNames += ( newName , ) # 새로운 이름 대입
            else :
                newNames += ( name , ) # 기존 이름 대입
        newName = newNames
    return
def nameDelete( ) :
    global names
    deleteName = input( 'deleteName : ')
    if names.count( deleteName ) == 0 : return
    else :
        newNames = ()               # 새로운 튜플
        for name in names :         # 튜플에서 하나씩 요소를 반복
            if name == deleteName : # 만약에 삭제할 요소값과 같으면
                continue            # 생략
            else :                  # 같지않으면
                newNames += ( name , ) # 새로운 튜플에 기존 요소를 누적
        names = newNames            # 반복문이 종료되면 새로운튜플을 전역변수에 대입
    return

while True :
    ch = int( input('1.create 2.read 3.update 4.delete : ') )
    if ch == 1 : nameCreate( )
    elif ch == 2 : nameRead( )
    elif ch == 3 : nameUpdate( )
    elif ch == 4 : nameDelete( )




