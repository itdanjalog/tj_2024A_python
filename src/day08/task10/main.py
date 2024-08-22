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
from region import Region
def service( ) :
    regionList = []
    f = open('인천광역시_부평구_인구현황.csv' , 'r')
    data = f.read()
    print( data )
    rows = data.split('\n')
    print( rows )
    rowCount = len( rows )
    for row in rows[1: rowCount-2] :
        print(row)
        cols = row.split(',')
        print( cols )
        region = Region( cols[0], int( cols[1] ),int( cols[2]),int( cols[3]),cols[4])
        print( region )
        regionList.append( region )
        print( regionList )
    # for region in regionList :
    #     print( region.info() )
    return regionList
