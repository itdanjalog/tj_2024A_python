# http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship
# 1. BeautifulSoup 이용한 쿠우쿠우 전국 매장 정보 크롤링
# 2. 전국 쿠우쿠우 매장 정보(번호,매장명,연락처,주소,영업시간)
# 3. pandas 이용한 csv 파일 로 변환
# 4. 플라스크 이용한 쿠우쿠우 전국 매장 정보 반환 하는 HTTP 매핑 정의한다.
    # HTTP(GET)  ip주소:5000/qooqoo
    # (3) 생성된 csv 파일 읽어서 json 형식을 반환
'''
1. URL 확인
    http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page=1
    http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page=2
    http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page=3
    ~~
    http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page=6
    [ 매개변수 파악 ] http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page={ 매개변수 }

2. 정보의 html 식별자 확인
    1. 정보가 있는 위치 <tbody> 여러개 매장 정보
    2. <tr> 하나의 매장 정보 # 홀수 또는 열 6개 :pc # 짝수 또는 열 1개 : 모바일
    3. <td> 하나의 매장의 각 속성  : <td> 1. 번호 2. 지점 3. 연락처 4. 주소 5. 영업시간
    4. 데이터 상세 위치 확인
        - 번호 <td>
        - 지점명 <td> -> <div> -> <a> 2번째
        - 연락처/주소/영엽시간 데이터는 <td> -> <a>
3. BeautifulSoup 이용한 구현
'''
# 1. 모듈 가져오기
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd

# [1] 쿠우쿠우 매장 정보 크롤링 서비스
def qooqooStoreInfo( result ) :
    for page in range( 1, 7 ) :
        # 2. 지정한 url 를 호출해서 응답 받기
        url = f"http://www.qooqoo.co.kr/bbs/board.php?bo_table=storeship&&page={page}"
        response = urllib.request.urlopen( url )
        if response.getcode() == 200 :
            print('>> 통신 성공')
            # 3. 통신 응답 결과를 읽어 와서 크롤링 파싱 준비
            soup = BeautifulSoup( response.read() , "html.parser" ); # print( soup )
            # 4. 분석한 HTML 식별자들을 파싱 , find , findall , select , select_one
            tbody = soup.select_one('tbody'); # print( tbody )  # 4-1 테이블 전체 파싱
            rows = tbody.select('tr'); # print(rows) # 4-2 테이블(전체매장) 마다 행(매장) 파싱
            for row in rows :  # 4-3 행(매장) 마다
                # print( row )
                cols = row.select('td');  # print( cols ) #  4-3 열(각매장정보) 파싱
                # 모바일 행 제외
                if len( cols ) <= 1 : # 만약에 열이 개수가 1개이면 모바일 이라고 가정해서 파싱 제외
                    continue # 가장 가까운 반복문으로 이동 , 아래 코드는 실행되지 않는다.
                # 각 정보들을 파싱 , *공백 제거
                번호 = cols[0].string.strip()
                지점명 = cols[1].select('a')[1].string.strip() # 2번째 열의 2개 a가 존재하는데 2번째 a태그의 텍스트를 파싱
                연락처 = cols[2].select_one('a').string.strip()
                주소 = cols[3].select_one('a').string.strip()
                영업시간 = cols[4].select_one('a').string.strip()

                # 5. 파싱한 정보를 리스트에 담기
                매장 = [ 번호 , 지점명 , 연락처 , 주소 , 영업시간 ] # print( 매장 ) # 리스트 에 담기 ( 왜? df 사용하기 위해서 2차원 리스트 구성할 예정 )
                result.append( 매장 ) # 리스트에 파싱한 리스트 담기 # 2차원 리스트
        else :
            print('>> 통신 실패 ')
    # 7. 리스트 반환
    return result

# [2] 2차원 리스트를 csv 변환해주는 서비스 , 데이터 , csv파일명 , 열(제목) 목록
def list2d_to_csv( result , fileName , colsNames ) :
    try :
        # 1. import pandas as pd  모듈 호출한다.
        # 2. 데이터프레임 객체 생성  , 데이터 , 열 목록
        df = pd.DataFrame( result , columns= colsNames )
        # 3. 데이터프레임 객체를 CSV 파일로 생성하기
        df.to_csv( f'{fileName}.csv' , encoding='utf-8' , mode='w')
        return True
    except Exception as e:
        print( e )
        return False

# 서비스 테스트 확인 구역
if __name__ == "__main__" :
    result = []
    qooqooStoreInfo( result ); print( result ); # 쿠우쿠우 매장 정보 크롤링 서비스 호출
    list2d_to_csv( result , '전국쿠우쿠우매장' , [ '번호','지점명','연락처','주소','영업시간'] )# 매장정보를 csv 로 저장 서비스 호출





















