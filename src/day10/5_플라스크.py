
# Flask 모듈 가져오기
from flask import Flask
import pandas as pd  # pandas 설치
import json

# Flask 객체 생성
app = Flask( __name__ )

# HTTP GET 매핑 설정
@app.route('/qooqoo')
def index() : # 매핑 함수
    df2 = pd.read_csv('쿠우쿠우.csv', encoding='utf-8', index_col=0, engine='python')
    # DataFrame을 JSON 문자열로 변환합니다
    json_result = df2.to_json(orient='records', force_ascii=False)
    print( json_result )
    json_result =  json.loads( json_result )
    # 변환된 JSON 문자열을 출력합니다
    return  json_result

# Flask 프레임워크 실행
if __name__ == '__main__' :
    app.run( host='0.0.0.0',debug=True )






