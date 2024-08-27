# day10 > 2_크롤링.py
from bs4 import BeautifulSoup # 모듈 가져오기
import urllib.request # 모듈 가져오기

# [실습1] http://quotes.toscrape.com
url = "http://quotes.toscrape.com" # 크롤링할 url
response = urllib.request.urlopen( url ) # 지정한 url 요청 후 응답 받기
htmlData = response.read() # 응답 받은 내용물 전체 읽어오기
# print( htmlData ) # 확인
soup = BeautifulSoup( htmlData , "html.parser") # 지정한 html문자열로 html 파싱객체 생성
# print( soup.prettify() )  # 확인

# 특정 마크업/태크 파싱
quoteDivs = soup.select('.quote')
# print( quoteDivs )
for quote in quoteDivs :
    # 각 명언 문구 만 추출
    print( quote.select_one('.text').string )
    # 각 명언 저자 추출
    print( quote.select_one('.author').string )
    # 각 명언 태그 목록 추출
    # print( quote.select('.tag') )
    for tag in quote.select('.tag') :
        print( tag.string , end = '\t' )
    print('\n\n\n')

# [실습2] https://v.daum.net/v/20240827074833139
url = 'https://v.daum.net/v/20240827074833139'
response = urllib.request.urlopen( url )
htmlData = response.read()
soup = BeautifulSoup( htmlData , "html.parser")
# print( soup ) # 확인

# 파싱하기
ps = soup.select( 'p' )
# print( ps )
for p in ps :
    #본문 내용 ( 사진 제외 )
    print( p.text )

#기사 제목
print( soup.select_one('.tit_view').string ) #
# print( soup.select_one('.news_view').text )

# [실습3]
    # https://search.naver.com/search.naver?query=부평구날씨
# https://search.naver.com/search.naver?query=%EB%B6%80%ED%8F%89%EA%B5%AC%EB%82%A0%EC%94%A8
url = "https://search.naver.com/search.naver?query="+urllib.parse.quote('부평구날씨') # URL 에 한글이 있을경우 : urllib.parse.quote( 한글 )
response = urllib.request.urlopen( url )
htmlData = response.read()
# print( htmlData )
soup = BeautifulSoup( htmlData , "html.parser" )
# print( soup )
# 온도 추출
print( soup.select_one('.temperature_text') )
# <div class="temperature_text"> <strong><span class="blind">현재 온도</span>27.2<span class="celsius">°</span></strong> </div>
print( soup.select_one('.temperature_text').text )
# 습도 추출
print( soup.select_one('.summary_list').select('.sort')[1].text )

























