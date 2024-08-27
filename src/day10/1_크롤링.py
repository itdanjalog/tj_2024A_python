# day10 > 1_크롤링.py

# 정적 웹 페이지 크롤링
'''
    [1] 설치
        from bs4 import BeautifulSoup
    [2] HTML 객체 파싱
        변수명 = BeautifulSoup( openhtml파일객체 , "html.parser" )
        변수명 = BeautifulSoup( html형식의문자열 , "html.parser" )
    [3] 파싱 메소드 , 속성
        1. .find( 식별자 , 속성명 = 값 )   : 지정한 식별자 과 속성명 동일한 태그 조회
            find( div )
            find( div , class_="box1" )
            find( div , id = "box2" )
        2. .select_one()
            select_one( div 또는 .box1 또는 #box2 )
        3. .findAll() : 지정한 식별자가 일치한 여러개 태그 추출해서 리스트로 반환
        4. .select()
        -
            .태그명    : 해당 태그들의 첫번째태그 추출
            .text     : 마크업 사이의 문자열 반환 , 자식 , 자손 가능 , 주로 중첩텍스트 일때
            .string   : 마크업 사이의 문자열 , 자식만 가능 , 자손 불가능 , 주로 단일텍스트 일떄
            .attrs   : 해당 마크업의 속성 목록/리스트 반환
'''
# [1] 설치
from bs4 import BeautifulSoup
'''
방법1 : from bs4 에 커서를 두고 빨간 느낌표 클릭후 -> beautifulsoup4 install
방법2 : 상단메뉴 -> 파일 -> 설정 -> 왼쪽메뉴[프로젝트] 하위 [인터프리터]
        -> [+] alt+insert -> beautifulsoup4 검색 후 패키지 선택 -> [패키지 설치]
'''
# [2] HTML 파일 객체
htmlFile = open( "1_웹페이지.html" , encoding="utf-8" )
# [3] BeautifulSoup 객체 생성
htmlObj = BeautifulSoup( htmlFile , "html.parser" )
print( htmlObj )
# [4] .find( 식별자 ) : 지정한 식별자의 마크업 조회하기  # .select_one( 식별자 )
print( htmlObj.find('div') ) # <div> [1] 여기를 크롤링 하세요.</div>
print( htmlObj.select_one('div') ) #<div> [1] 여기를 크롤링 하세요.</div>
# [5] .findAll( 식별자 ) : 지정한 식별자의 마크업 여러개 조회하기 # .select( 식별자 )
print( htmlObj.findAll( 'div') ) # [<div> [1] 여기를 크롤링 하세요.</div>, <div class="box1"> [2] 여기를 크롤링 하세요.</div>, <div id="box2"> [3] 여기를 크롤링 하세요. </div>]
print( htmlObj.select('div') ) # [<div> [1] 여기를 크롤링 하세요.</div>, <div class="box1"> [2] 여기를 크롤링 하세요.</div>, <div id="box2"> [3] 여기를 크롤링 하세요. </div>]
# [6] .text : 호출된 마크업의 있는 내용물을  문자열 추출 # .string
print( htmlObj.find('div').text ) # [1] 여기를 크롤링 하세요.
print( htmlObj.find('div').string )  # [1] 여기를 크롤링 하세요.
# [7] 반복문과 같이 활용
for div in htmlObj.select('div') : # 모든 div 를 추출해서 리스트 반환 한 다음 리스트 만큼 반복문 처리
    print( div.string ) # div 하나씩 내용물 추출
# [8] class 식별자 이용한 조회
print( htmlObj.find('box1') ) # None
print( htmlObj.find('.box1') ) # None
print( htmlObj.find('div', class_="box1") ) # <div class="box1"> [2] 여기를 크롤링 하세요.</div>
print( htmlObj.select_one('.box1') ) #<div class="box1"> [2] 여기를 크롤링 하세요.</div>
# [9] id 식별자 이용한 조회
print( htmlObj.find('div' , id = 'box2' ) ) # <div id="box2"> [3] 여기를 크롤링 하세요. </div>
print( htmlObj.select_one('#box2')) # <div id="box2"> [3] 여기를 크롤링 하세요. </div>

# 연습
html = '''
    <h1 id="title">한빛출판네트워크</h1>
    <div class="top">
        <ul class="menu">
            <li><a href="http://wwww.hanbit.co.kr/member/login.html"class="login">로그인</a>
            </li>
        </ul>
        <ul class="brand">
            <li><a href="http://www.hanbit.co.kr/media/">한빛미디어</a></li>
            <li><a href="http://www.hanbit.co.kr/academy/">한빛아카데미</a></li>
        </ul>
    </div>'''
# [1] html 파싱 객체
soup = BeautifulSoup( html  , 'html.parser' )
print( soup )
print( soup.prettify() ) # HTML 문서 형태로 출력 해주는 함수
# [2] 태그(마크업) 파싱하기
print( soup.h1 ) # 1. 파싱객체.마크업명  # <h1 id="title">한빛출판네트워크</h1>
print( soup.div ) # <div class="top"> ~~ </div>
print( soup.ul ) # <ul class="menu"> ~~ </ul>
print( soup.li ) # <li> ~~ </li>
print( soup.a) # <a class="login" href="http://wwww.hanbit.co.kr/member/login.html"> ~~ </a>
print( soup.findAll( "ul" ) ) #
print( soup.findAll( 'li' )) #
print( soup.findAll( 'a')) #
# .attrs : 지정한 마크업의 속성 목록을 딕셔너리 반환
print( soup.findAll( 'a')[0].attrs ) # {'href': 'http://wwww.hanbit.co.kr/member/login.html', 'class': ['login'] }
print( soup.findAll( 'a')[0]['href'] ) # http://wwww.hanbit.co.kr/member/login.html
print( soup.findAll( 'a')[0]['class'] ) # ['login']
print( soup.find( 'ul' , attrs={ 'class' : 'brand'})) # vs find( 마크업 , class_='클래스명' )
print( soup.find( id="title") ) # vs .select_one(#ID명)
li_list = soup.select( 'div > ul.brand>li ')
for li in li_list :
    print( li.string )


















