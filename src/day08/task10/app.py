from flask import Flask, jsonify

from main import service

from flask_cors import CORS

app = Flask(__name__)

# 모든 경로에 대해 CORS 허용
CORS(app)

@app.route('/', methods=['GET'])
def get_data():
    # 예시 데이터
    data = service()
    data = list(map(lambda o: o.__dict__, data))
    return data

if __name__ == '__main__':
    app.run(debug=True)