# HTTP 매핑 정의 # 삼성전자 csv 파일 읽어오는 함수
# 1. Flask 객체 필요 , 다른 파일 존재 , 모듈 호출
from app import app # from 파일명.py import 변수 또는 함수 또는 클래스 또는 *
# 2. Flask 객체를 이용한 HTTP 매핑 정의 # [GET] http://localhost:5000/samsung
@app.route("/samsung" , methods = ["get"] )
def samSungInfo() :
    response = []
    f = open("삼성전자주가.csv" , "r") # (1) 파일 읽기 모드
    data = f.read() # (2) 파일 읽기
    rows = data.split("\n") # (3) 행 구분
    for row in rows[1:] : # (4) 첫 줄 제외한 행 반복
        cols = row.split(',') #(5) 열 구분
        dic = { # 일자,종가,대비,등락률,시가,고가,저가,거래량,거래대금,시가총액,상장주식수
            '일자' : eval( cols[0] ) ,
            '종가': format( int( eval( cols[1] ) ) , ',' ) , # 천단위 쉼표 : format( 숫자데이터 , ',' )
            '대비': eval( cols[2] ) ,
            '등락률': eval( cols[3] ),
            '시가': eval( cols[4]),
            '고가': eval( cols[5]),
            '저가': eval( cols[6]),
            '거래량': eval( cols[7]),
            '거래대금': eval( cols[8]),
            '시가총액': eval( cols[9]),
            '상장주식수': eval( cols[9])}# (6) 딕셔너리 구성
        response.append( dic ) # (7) 리스트에 구성한 딕셔너리 담기
    return response # (8) 구성된 딕셔너리 리스트 반환

# print( samSungInfo() ) # 확인



