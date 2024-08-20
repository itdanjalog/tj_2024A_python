# day05 > Task2.py
# 문자열 활용 , p.50 ~ p.76
# [조건1] : 각 함수 들을 구현 해서 프로그램 완성하기
# [조건2] :  1. 하나의 이름을 입력받아 names 에 저장합니다.
#           2. 저장된 여러명의 이름들 names 을 모두 출력 합니다.
#           3. 수정할 이름을 입력받아 존재하면 새로운 이름을 입력받고 수정 합니다.
#           4. 삭제할 이름을 입력받아 존재하면 삭제 합니다.
# [조건3] : names 변수 외 추가적인 전역 변수 생성 불가능합니다.
# [조건4] : 최대한 리스트타입 사용하지 않기.
# 제출 : git에 commit 후 카톡방에 해당 과제가 있는 링크 제출

names = "" # 여러개 name들을 저장하는 문자열 변수
def nameCreate( ) :
    return
def nameRead( ) :
    return
def nameUpdate( ) :
    return
def nameDelete( ) :
    return
while True : # 무한루프
    ch = input("1.create 2.read 3.update 4.delete : ")
    if ch == 1 :
        nameCreate()
    elif ch == 2 :
        nameRead()
    elif ch == 3 :
        nameUpdate()
    elif ch == 4 :
        nameDelete()

