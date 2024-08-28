# 플라스크 : 파이썬 이용한 경량 웹 프레임워크
# 1. 플라스크 모듈 가져오기
from flask import Flask
# 2. 플라스크 객체 생성
app = Flask( __name__ )
# 3. CORS 허용 , 서로 다른 port 간의 데이터 통신 허용 , 보안상 문제 발생 할수 있다.
from flask_cors import CORS
CORS(app) # 모든 HTTP 에 대해 CORS 허용

# 4. controller 모듈 가져오기
from controller import *

# 3. 플라스크 웹 실행 , 1과2번3번 코드 고정
if __name__ == '__main__' :
    app.run( host='0.0.0.0' , debug=True )
    # http://127.0.0.1:5000 # http://localhost:5000 # http://192.168.30.9:5000
# ---- 서버 재실행시 주의할점 : 직접 현재 실행 중지하고 다시 실행