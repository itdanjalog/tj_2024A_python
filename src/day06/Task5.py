# day06 > Task5.py
# 딕셔너리/리스트 활용 , 파일처리 , p.175 ~ p.182
# [조건1] : 각 함수 들을 구현 해서 프로그램 완성하기
# [조건2] :  1. 한명의 name , age 를 입력받아 저장 합니다.
#           2. 저장된 여러명의 name , age 을 모두 출력 합니다.
#           3. 수정할 이름을 입력받아 존재하면 새로운 name , age 을 입력받고 수정 합니다.
#           4. 삭제할 이름을 입력받아 존재하면 삭제 합니다.
# [조건3] : names 변수 외 추가적인 전역 변수 생성 불가능합니다.
# [조건4] : 프로그램이 종료되고 다시 실행되더라도 기존의 names 데이터가 유지 되도록 파일처리 하시오.
#           - dataLoad() , dataSave() 함수를 정의하여 적절한 위치 에서 호출 하시오.
# 제출 : git에 commit 후 카톡방에 해당 과제가 있는 링크 제출
names = [ ]
def dataLoad( ) : # 파일내 데이터를 불러오기
    global names
    return
def dataSave( ) :  # 데이터를 파일내 저장하기
    global names
    return
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
while True :
    ch = int( input('1.create 2.read 3.update 4.delete : ') )
    if ch == 1 : nameCreate( )
    elif ch == 2 : nameRead( )
    elif ch == 3 : nameUpdate( )
    elif ch == 4 : nameDelete( )