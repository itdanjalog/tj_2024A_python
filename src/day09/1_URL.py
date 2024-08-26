# day09 > 1_URL_py
# urllib : URL 작업을 위한 여러 모듈을 모은 패키지

# [1] urllib.request : URL 요청에 관련 기능을 제공하는 라이브러리
    # 1. urllib.request.Request( URL ) : 지정한 URL에 대한 요청 객체를 반환
    # 2. urllib.request.urlopen( 요청객체 ) : 지정한 요청 객체를 실행하고 응답 객체를 반환
    #   2-1 응답객체.getcode() : 응답 상태 반환 ( 2xx : 성공  4xx : 실패 5xx :실패 )
    #   2-2 응답객체.read() : 응답 내용 모두 읽어오기 # .decode('utf-8') : HTML 형식과 한글 형식 지원
import urllib.request # 1. request 모듈 호출
요청할주소 = "https://www.example.com" # 2. 요청을 보낼 url 주소를 가지는 변수
요청객체 = urllib.request.Request( 요청할주소 ) # 3. Request 객체 생성 # 지정한 URL에 대한 요청을 생성
응답객체 = urllib.request.urlopen( 요청객체 ) # 4. urlopen 메소드를 이용한 url에 대한 요청을 실행하고 응답 객체를 반환 함수
print( 응답객체 ) # <http.client.HTTPResponse object at 0x0000018A884FC910>
print( 응답객체.getcode() ) # 200
#print( 응답객체.read() ) # b'<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>\n\n    <meta charset="utf-8" />\n    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <style type="text/css">\n    body {\n        background-color: #f0f0f2;\n        margin: 0;\n        padding: 0;\n        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;\n        \n    }\n    div {\n        width: 600px;\n        margin: 5em auto;\n        padding: 2em;\n        background-color: #fdfdff;\n        border-radius: 0.5em;\n        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);\n    }\n    a:link, a:visited {\n        color: #38488f;\n        text-decoration: none;\n    }\n    @media (max-width: 700px) {\n        div {\n            margin: 0 auto;\n            width: auto;\n        }\n    }\n    </style>    \n</head>\n\n<body>\n<div>\n    <h1>Example Domain</h1>\n    <p>This domain is for use in illustrative examples in documents. You may use this\n    domain in literature without prior coordination or asking for permission.</p>\n    <p><a href="https://www.iana.org/domains/example">More information...</a></p>\n</div>\n</body>\n</html>\n'
print( 응답객체.read().decode('utf-8'))