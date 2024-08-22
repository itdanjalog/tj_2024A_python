# day07 > task8 > file.py
from user import User

# 1. save
def save( names ) :
    # 1. 쓰기 모드 파일 객체
    f = open( 'user.txt' , 'w' , encoding='utf-8')
    # 2. 내용구성
    outstr = ''
    for user in names : # 객체를 문자열 변환
        outstr += f'{user.name},{user.age}\n'  # CSV 형식
    f.write( outstr )   # 3. 파일 쓰기
    f.close()  # 4. 파일닫기
    return
# 2. load
def load( ) :
    #1. 읽기 모드 파일 객체
    try: # 예외 처리 # 예외가 발생 할것 같은 코드
        f = open( 'user.txt' , 'r' , encoding='utf-8')
        #2. 읽어오기
        names = [] # 읽어온 내용들의 객체들을 저장하는 리스트
        data = f.read()
        rows = data.split("\n")             # 행 구분 \n
        for row in rows :
            if row : # 만약에 데이터가 존재하면
                cols = row.split(',')       # 열 구분 ,(쉼표)
                user = User( cols[0] , cols[1] )
                names.append( user )
        f.close() #3. 파일 닫기
        return names # 4. 리스트 반환

    except FileNotFoundError : # 예외 처리 # 예외가 발생 했을때 실행되는 구역
        return [] # 빈 리스트  반환

