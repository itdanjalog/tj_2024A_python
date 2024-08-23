# day08 > 3_FlaskCors.py
'''
    CORS : 교차 출처 자원 공유
        - Cross - origin Resource Sharing
        - 현재 주소에서 다른 주소로부터 통신요청 하고 현재 주소를 응답 받기

    HTTP 통신
        (유재석) 안녕 ----요청---->  (유재석)
                    <---응답-----
        :8080 ---------------------->   :8080
        :8080 <----------------------   :8080
    CORS 통신 제한
        (유재석) 안녕 ----요청---->   (강호동)
                    <---응답-----
        8080(:5000) ----------------------> :5000
        8080  <---------------------- : 8080
    CORS 허용
        (1) Flask-Cors 설치
                - 상단메뉴 -> 파일 -> 설정 -> 왼쪽메뉴[프로젝트] 하위 [인터프리터]
                    -> [+] alt+insert -> Flask-Cors 검색 후 패키지 선택 -> [패키지 설치]
        (2) 모듈 가져오기
            from flask_cors import CORS
        (3) CORS 허용
            CORS( app )
'''
from flask import Flask
app = Flask( __name__ )

from flask_cors import CORS
CORS( app ) # 모든 경로에 대해 CORS 허용

@app.route( "/" )
def index() :
    return "Hello Python Flask CORS"

if __name__ == "__main__" :
    app.run( )


