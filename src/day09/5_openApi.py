# day09 > 5_openApi.py
# 연습문제 : 서울 열린데이터 광장에서 '상상대로 서울 자유제안 정보' 을 크롤링하여 JSON파일로 저장하시오.
# https://data.seoul.go.kr/dataList/OA-2563/S/1/datasetView.do

# 회원가입/로그인 후 인증키 발급 받기
# http://openapi.seoul.go.kr:8088/(인증키)/json/ChunmanFreeSuggestions/시작번호/끝번호

# 강사 인증키 : 516a52754c717765383054574f7273
# 예시]
# http://openapi.seoul.go.kr:8088/516a52754c717765383054574f7273/json/ChunmanFreeSuggestions/1/1000
# http://openapi.seoul.go.kr:8088/516a52754c717765383054574f7273/json/ChunmanFreeSuggestions/1001/2000
# http://openapi.seoul.go.kr:8088/516a52754c717765383054574f7273/json/ChunmanFreeSuggestions/2001/3000

