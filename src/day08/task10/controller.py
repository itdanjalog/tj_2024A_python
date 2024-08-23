#day08 > task10 > controller.py
from app import app # Flask 객체 호출
from service import personData # 서비스 함수 가져오기

@app.route("/" , methods = ['GET'] )
def index() :
    data = personData() # 서비스 처리후 결과 받기
    # 파이썬 객체로는 JSON 전송이 불가능.
    # 파이썬 객체를 딕셔너리 로 변경 ,  # 객체명.__dict__
    newData = [ ]
    for o in data :
        dic = o.__dict__
        newData.append( dic )
    return newData

