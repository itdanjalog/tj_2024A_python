# day10 > 5_쿠우쿠우매장.py
# http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship
# 1. BeautifulSoup 이용한 쿠우쿠우 전국 매장 정보 크롤링
# 2. 전국 쿠우쿠우 매장 정보(번호,매장명,연락처,주소,영업시간)
# 3. pandas 이용한 csv 파일 로 변환
# 4. 플라스크 이용한 쿠우쿠우 전국 매장 정보 반환 하는 HTTP 매핑 정의한다.
    # HTTP(GET)  ip주소:5000/qooqoo
    # (3) 생성된 csv 파일 읽어서 json 형식을 반환

from bs4 import BeautifulSoup
import urllib.request
import pandas as pd  # pandas 설치

result = []
for page in range(1, 7):  # 1 ~ 50까지 반복
    url = f"http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page={page}"  # 할리스 매장 정보 url
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup( response, "html.parser");  # print( soup )
    tbody = soup.select_one('tbody'); # print( tbody )
    for row in tbody.select('tr'):
        tds = row.select('td')
        if len(tds) <= 3 : continue
        store_sido = tds[1].select('a')[1].string.lstrip();  print( store_sido )
        store_name = tds[2].select('a')[0].string.lstrip();  print( store_name )
        store_address = tds[3].select('a')[0].string.lstrip();  print( store_address )
        store_phone = tds[4].select('a')[0].string.lstrip();  print( store_phone )
        result.append( [ store_sido,store_name,store_address ,store_phone] )

# print( result )
df = pd.DataFrame( result , columns=[  'store' , 'sido-gu' , 'address','phone' ])
# print( df )
df.to_csv('쿠우쿠우.csv', encoding="utf-8", mode='w' )