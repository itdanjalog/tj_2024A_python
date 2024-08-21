# day06 > Task3.py
# 딕셔너리/리스트 활용 , p.93 ~ p.101
# [조건1] : 각 함수 들을 구현 해서 프로그램 완성하기
# [조건2] :  1. 하나의 이름을 입력받아 names 에 저장합니다.
#           2. 저장된 여러명의 이름들 names 을 모두 출력 합니다.
#           3. 수정할 이름을 입력받아 존재하면 새로운 이름을 입력받고 수정 합니다.
#           4. 삭제할 이름을 입력받아 존재하면 삭제 합니다.
# [조건3] : names 변수 외 추가적인 전역 변수 생성 불가능합니다.
# [조건4] : 최대한 리스트타입 사용하지 않기.
# 제출 : git에 commit 후 카톡방에 해당 과제가 있는 링크 제출

    # 딕셔너리 란? { } 안에 key : value 쌍으로 저장하는 자료 , JSON 비슷하다.
    # 리스트[ ] 안에 여러개 딕셔너리{ } 저장된 구조
names = [ { 'name' : '유재석' } ,
    { 'name' : '강호동' } ,
    { 'name' : '신동엽'} ] # 샘플 데이터

def nameCreate( ) :
    global names
    newName = input('newName : ')
    dic = { 'name' : newName }  # 딕셔너리 구성
    names.append( dic ) # 딕셔너리를 리스트에 삽입
    return

def nameRead( ) :
    global names
    for dic in names : # 리스트내 딕셔너리 하나씩 호출
        print( f'name : { dic['name'] }' )  # 딕셔너리변수명[key] 또는 딕셔너리변수명.get( key )
    return


def nameUpdate(  ) :
    global names
    oldName = input('oldName : ')
    for dic in names :
        if dic['name'] == oldName :
            newName = input('newName : ')
            dic['name'] = newName # 해당 딕셔너리의 속성 값 수정 하기.
            return
    return

def nameDelete( ) :
    global names
    deleteName = input('deleteName : ')
    for dic in names :
        if dic['name'] == deleteName : #만약에 삭제할 이름과 같으면
            names.remove( dic ) # 리스트변수명.remove( 삭제할딕셔너리 )
            return  # 1개만 삭제하기 위해서는 삭제후 return
    return

while True :
    ch = int( input('1.create 2.read 3.update 4.delete : ') )
    if ch == 1 : nameCreate( )
    elif ch == 2 : nameRead( )
    elif ch == 3 : nameUpdate( )
    elif ch == 4 : nameDelete( )




