# day08 > 2_FlaskHttp.py
from flask import Flask #(1)
app = Flask( __name__ ) #(2)

class Test :
    pass
# =========== HTTP 매핑 =========== #
# @app.route( "HTTP경로정의" , methods = [ 'HTTP method' ]
# return : Flask 에서 HTTP Response Content-Type : 파이썬의 리스트 타입 , 딕셔너리 타입 , 문자열 타입( JSON ) 제공
@app.route( "/" , methods = ['GET']) # vs spring @GetMapping("HTTP경로정의" )
def index1( ) :
    return "Hello HTTP method GET" # HTTP Response ContentType : Content-Type:	text/html;

@app.route("/" , methods =  ['POST'] ) # vs spring @PostMapping("HTTP경로정의" )
def index2( ) :
    # return True # 타입 오류  ,  flask 에서 파이썬 논리타입 HTTP 응답이 불가능하다.
    # return 3    # 타입 오류  ,  flask 에서 파이썬 정수타입 HTTP 응답이 불가능하다.
    return [ 3, 3 ] # HTTP Response ContentType : Content-Type:	application/json;       # vs spring @ResponseBody

@app.route("/" , methods =  ['PUT'] ) # vs spring @PutMapping("HTTP경로정의" )
def index3( ) :
    return { 'result' : True } # HTTP Response ContentType : Content-Type:	application/json;

@app.route("/" , methods =  ['DELETE'] ) # vs spring @DeleteMapping("HTTP경로정의" )
def index4( ) :
    # return Test() # 타입 오류  ,  flask 에서 파이썬 객체로 HTTP 응답이 불가능하다.
    return "true"

# ================================ #
if __name__ == "__main__" : #(3)
    app.run( ) # debug=True 디버그[정보또는오류 콘솔출력 제공] 모드


