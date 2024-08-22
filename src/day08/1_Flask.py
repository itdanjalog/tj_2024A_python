from flask import Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_data():
    # 예시 데이터
    return { 'data' : 1 }

if __name__ == '__main__':
    app.run(debug=True)