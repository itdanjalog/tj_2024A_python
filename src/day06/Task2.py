# day06 > Task2.py
# 리스트 활용 , p.77 ~ p.88
# [조건1] : 각 함수 들을 구현 해서 프로그램 완성하기
# [조건2] :  1. 하나의 이름을 입력받아 names 에 저장합니다.
#           2. 저장된 여러명의 이름들 names 을 모두 출력 합니다.
#           3. 수정할 이름을 입력받아 존재하면 새로운 이름을 입력받고 수정 합니다.
#           4. 삭제할 이름을 입력받아 존재하면 삭제 합니다.
# [조건3] : names 변수 외 추가적인 전역 변수 생성 불가능합니다.
# [조건4] : 최대한 리스트타입 사용하지 않기.
# 제출 : git에 commit 후 카톡방에 해당 과제가 있는 링크 제출


    # 리스트란? 여러개 자료들을 인덱스로 구분하여 하나의 리스트 자료 구성
names = [ '유재석' , '강호동' , '신동엽' ] # 샘플 데이터

def nameCreate( ) :
    global names
    newName = input('newName : ')
    names.append( newName ) # 리스트변수.append( 새로운값 ) : 리스트내 마지막 요소 뒤에 새로운값 요소 추가
    return

def nameRead( ) :
    global names
    for name in names :
        print( f'name : { name }')
    return

def nameUpdate(  ) :
    global names
    oldName = input('oldName : ')
    #만약에 수정할 이름이 존재하지 않으면
    if names.count( oldName ) == 0 : return
    else :  # 존재하면 # 수정할 이름의 인덱스 찾기  ,
        # 리스트변수명.index( 찾을값 ) : 리스트내 찾을값이 존재하면 인덱스 반환
        index = names.index( oldName )
        newName = input('newName : ') # 새로운 이름
        names[ index ] = newName # 찾은 인덱스에 새로운 값 대입
    return

def nameDelete( ) :
    global names
    deleteName = input('deleteName : ')
    if names.count( deleteName ) == 0 : return
    else : # 삭제할 값이 존재하면
        names.remove( deleteName ) # 리스트변수명.remove( 삭제할값 )
    return

while True :
    ch = int( input('1.create 2.read 3.update 4.delete : ') )
    if ch == 1 : nameCreate( )
    elif ch == 2 : nameRead( )
    elif ch == 3 : nameUpdate( )
    elif ch == 4 : nameDelete( )




