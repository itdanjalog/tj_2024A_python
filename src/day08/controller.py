
from app import app  # app 객체를 import

@app.route('/', methods=['POST'])
def get_data1():
    # 예시 데이터
    return { 'data' : 1 }


@app.route('/', methods=['GET'])
def get_data2():
    # 예시 데이터
    return { 'data' : 2 }


@app.route('/', methods=['PUT'])
def get_data3():
    # 예시 데이터
    return { 'data' : 3 }

@app.route('/', methods=['DELETE'])
def get_data4():
    # 예시 데이터
    return { 'data' : 4 }