# 주제 : 다음 경제 뉴스의 최신 10페이지 기사들 제목의 단어 빈도수 분석
# https://news.daum.net/breakingnews/economic?page=1 ~ 10
import re
import urllib.request
from bs4 import BeautifulSoup

# 1. 데이터 준비
textData = []
    # [크롤링]
for page in range( 1 , 11 ) : # 1부터 10까지 반복처리
    url = f'https://news.daum.net/breakingnews/economic?page={ page }'  # 1. 크롤링 url 준비
    response = urllib.request.urlopen( url ) # 2. 지정한 url 열고 응답 받기
    soup = BeautifulSoup( response , "html.parser") # 3. 응답 결과를 html로 파싱
    list_news2 = soup.select_one( '.list_news2') # 4.특정 클래스 파싱
    for li in list_news2.select('li') : # 5.특정 클래스 파싱 해서 반복문 돌리기
        title = li.select_one('.tit_thumb > a').string # 6.특정 클래스 파싱
        textData.append( title ) # 7. 기사 제목을 리스트에 담기
# 2. 품사 태깅
    # 1. 정규표현식  # 단어가 아닌 특수문자를 공백으로 치환 해서 하나의 문자열로 담기
message = ''
for msg in textData :
    message += re.sub( r'[^\w]' , ' ' , msg )
    # 2. 품사 태깅
from konlpy.tag import Okt
okt = Okt()
words = okt.nouns( message )

# 3. 분석( 빈도수 )
    # 1. 빈도수 분석
from collections import Counter
wordsCount = Counter( words )
    # 2. 상위 N개 추출
bestWords = wordsCount.most_common( 30 )
    # 3. 딕셔너리 변환
wordsDict = { }
for word , count in bestWords :
    if len( word ) > 1 :
        wordsDict[word] = count

# 4. 시각화 ( 히스토그램 , 워드클라우드 )

# 히스토그램
import matplotlib.pyplot as plt
from matplotlib import font_manager , rc # 맷플롯립 에서 폰트 관련 객체
import matplotlib
# 맷플롯립 에서 한글화
# 3-1 시스템(컴퓨터) 폰트의 경로가 명확 하지 않을때
    # (1) 폰트 경로
font_path = "c:/windows/fonts/malgun.ttf" # 윈도우 기준의 폰트가 설치된 경로 : c:/windows/fonts/폰트명
    # (2) 폰트 설정 객체를 이용한 폰트 설정
font_name = font_manager.FontProperties( fname= font_path ).get_name() # 지정한 경로의 폰트 파일에서 폰트 이름을 가져옴
    # (3) 차트에 적용
matplotlib.rc( 'font' , family = font_name )

plt.bar( wordsDict.keys() , wordsDict.values() ) # plt.bar( x축값 , y축값 )
plt.xticks( rotation = 75 ) # x축 레이블 기울기
plt.show() # 차트 열기


# 워드클라우드
from wordcloud import WordCloud
import matplotlib.pyplot as plt

font_path = "c:/windows/fonts/malgun.ttf"
wc = WordCloud( font_path ).generate_from_frequencies( wordsDict )

plt.imshow( wc )
plt.show()



























