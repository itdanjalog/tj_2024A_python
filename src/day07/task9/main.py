# day07 > task9 > main.py
'''
    csv파일 다루기
    파일 : 인천광역시_부평구_인구현황.csv
    [조건1] 부평구의 동 별(마다) Region 객체 생성 해서 리스트 담기
    [조건2]
        Region 객체 변수 :
            1.동이름 2.총인구수 3.남인구수 4.여인구수 5.세대수
        Region 메서드
            1. 남자 비율 계산 함수
            2. 여자 비율 계산 함수
    [조건3] 모든 객체의 정보를 f포메팅 해서 console 창에 출력하시오.
    [조건4] 출력시 동 마다 남 여 비율 계산해서 백분율로 출력하시오.
    [출력예시]
        부평1동       35141,  16835,  18306,  16861   59%     41%
        부평2동       14702,  7289,   7413,   7312    51%     49%
        ~~~~~~
'''

# [모듈] from py파일명 import 해당파일에호출할식별자
from region import Region
if __name__ == "__main__" :
    try: # 예외처리
        regionList = [ ]
        # (1) 파일 읽기모드
        f = open( '인천광역시_부평구_인구현황.csv' , 'r')
        # (2) 파일 전체 읽어오기
        data = f.read(); print( data )
        # (3) 데이터 가공 ( csv 형식 ) , 행마다 분리
        rows = data.split('\n'); print( rows )
        # (4) 행마다 반복문 , 첫줄 , 뒤에 2줄 제외
        rowCount = len( rows ) # 데이터의 전체 행 개수
        for row in rows[ 1 : rowCount-2 ] :
            print( row )
            # (5) 열마다 분리
            cols = row.split(','); print( cols )
            # (6) 해당 열 들을 객체화 , 타입 변환
            region = Region( cols[0] , int( cols[1] ) ,
                             int( cols[2] ) , int( cols[3]) ,
                             int(cols[4] ) ); print( region )
            # (7) 리스트 담기
            regionList.append( region ); print( regionList )
        # (8) 리스트내 객체 정보 호출
        for region in regionList :
            print( region.toString() )
    # 예외가 발생했을때
    except FileNotFoundError as e : print( e )
    except Exception as e : print( e )


























