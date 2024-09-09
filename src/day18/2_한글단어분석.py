# day18 -> 2_한글단어분석.py
# 자료 준비 : etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명.json

# 1. 데이터준비
# 1-1 파일 읽기
fileName = 'etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명.json'
file = open( fileName , 'r' , encoding='utf-8')
fileData = file.read( )
import json
jsonData = json.loads( fileData )
# json.loads() : JSON형식 ---> PY형식 으로 변환
# json.dumps() : PY형식 ---> JSON형식 으로 변환
print( jsonData )

