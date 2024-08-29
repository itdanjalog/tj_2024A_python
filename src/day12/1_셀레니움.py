# day12 > 셀레니움.py
# 1. 모듈 가져오기
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import pandas as pd

'''
[설치방법]
방법1 : from selenium 에 커서를 두고 빨간 느낌표 클릭후 -> install selenium
방법2 : 상단메뉴 -> 파일 -> 설정 -> 왼쪽메뉴[프로젝트] 하위 [인터프리터]
        -> [+] alt+insert -> selenium 검색 후 패키지 선택 -> [패키지 설치]
'''
# [1] 커피빈 매장 정적 크롤링 함수
def CoffeBean_store( result ) :
    # 2. webdriver 객체 생성
    wd = webdriver.Chrome()

    # 3. webdriver 객체를 이용한 웹페이지 접속 , .get( URL )
    # wd.get( "http://www.hanbit.co.kr")
    for i in range( 1, 10 ) : # 테스트는 1~9 까지 만 확인
        # 실습 : 커피빈 매장 정보(동적페이지=JS이벤트) 크롤링
        # 1. 커피빈 웹페이지 연결
        wd.get('https://www.coffeebeankorea.com/store/store.asp')
        time.sleep(2) # 2초 일시정지(대기상태) # 웹페이지가 열릴때까지 2초 대기 # 컴퓨터 사양에 따라 다를수 있다

        try : # 예외처리
            # 2. 커피빈 웹페이지의 자바스크립트 함수 호출 ,  .execute_script("JS함수호출")
            # wd.execute_script("storePop2(31)") # 31번 매장인 '삼성봉은사거리점' 모달 창 열린다.
            wd.execute_script( f'storePop2({ i })')
            time.sleep(2) # 2초 일시정지(대기상태) # 스크립트가 실행이 끝날때까지 2초 대기 # 컴퓨터 사양에 따라 다를수 있다

            # 3. 자바스크립트 함수가 수행된 페이지의 소스 코드를 저장
            html = wd.page_source
            # 4. BeautifulSoup 객체 생성
            soupCB1 = BeautifulSoup( html , 'html.parser')
            # print( soupCB1.prettify() ) # 5. HTML 소스 코드 확인

            # 6. 특정 매장 정보의 모달창에서 매장 정보 파싱하기
            # 6-1 매장 지점명
            store_name_h2 = soupCB1.select('div.store_txt > h2 ') # print( store_name_h2 ) # [<h2>삼성봉은사거리점</h2>]
            store_name = store_name_h2[0].string #print( store_name ) # <h2>삼성봉은사거리점</h2>
            # 6-2 매장 주소
            store_info = soupCB1.select('div.store_txt > table.store_table > tbody > tr > td ') # print( store_info )
            # [<td> 평일 07:00-22:30 | 주말/공휴일 07:00-22:00</td>, <td>건물뒷편주차장(기계주차식)<br/>평일 최초 30분 2,000원 / 1시간 5,000원 (구매영수증 지참시 30분 2,000원 / 1시간 3,500원)<br/>1시간 초과시 10분당 1,000원 (평일,주말 동일)</td>, <td>서울시 강남구 영동대로 607 1,2층  <!--span class="lot">(서울시 강남구 영동대로 607 1,2층)</span--></td>, <td>02-3443-5618</td>, <td class="best">식약처 인증 위생등급 매우 우수매장</td>, <td class="hallcake">홀케익 당일 수령가능</td>]
            store_address_list = list( store_info[2] ) # print( store_address_list )
            # ['서울시 강남구 영동대로 607 1,2층  ', 'span class="lot">(서울시 강남구 영동대로 607 1,2층)</span']
            store_address = store_address_list[0] # print( store_address )
            # 6-3 매장 전화번호
            store_phone = store_info[3].string # print( store_phone )
            # 매장 정보의 리스트 선언
            store = [ store_name , store_address , store_phone ]
            # 매장 정보 리스트를 매장리스트에 담기 , 2차원 리스트
            result.append( store )

        except Exception as e : print( e )

    # for end
    return result

def main() :
    result = [ ]
    CoffeBean_store( result ); print( result )
    # 판다스를 이용한 2차원 리스트를 데이터 프레임 객체 로 생성
    CB_tbl = pd.DataFrame( result , columns=['store','address','phone'] )
    # 데이터프레임 객체 정보를 csv 파일로 저장
    CB_tbl.to_csv( 'coffeBean.csv' , encoding='utf-8' , mode='w' , index=True )

if __name__ == '__main__' :
    main()















