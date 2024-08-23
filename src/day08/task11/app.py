# day08 > task11 > app.py
'''
    삼성전자주가.csv 파일의 정보를 테이블 형식으로 localhost:8080/index3.html 출력하시오.
        1. csv 파일을 읽어서 한줄씩 딕셔너리 로 변환후 리스트 담기
        2. 플라스크 이용한 HTTP 매핑 정의 하기
        3. 스프링 서버에서 AJAX 를 이용한 플라스크 서버로 부터 삼성전자주가 정보 응답받기
'''
from flask import Flask #(1) Flask 모듈 가져오기
app = Flask(__name__)#(2) Flask 객체 생성
from flask_cors import CORS #(3) CORS 모듈 가져오기  # CORS : 교차 출처 자원 공유 허용
CORS( app ) # (4) 해당 Flask 객체내 모든 HTTP 에 대해 CORS 허용

#(6) HTTP 매핑 들이 있는 파일 모듈 가져오기
from controller import * # 파일내 모두 가져오기

if __name__ == "__main__" :
    app.run()   # (5) Flask 실행 #http://127.0.0.1:5000 #http://localhost:5000 # http://192.168.30.9:5000



