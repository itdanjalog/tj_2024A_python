#day08 > task10 > service.py

from region import Region
def personData() :
    regionList = [ ]
    f = open( '인천광역시_부평구_인구현황.csv' , 'r')
    data = f.read()
    rows = data.split('\n')
    rowCount = len( rows )
    for row in rows[ 1 : rowCount-2 ] :
        cols = row.split(',');
        region = Region( cols[0] , int( cols[1] ) ,
                         int( cols[2] ) , int( cols[3]) ,
                         int(cols[4] ) ); print( region )
        regionList.append( region ); print( regionList )

    return regionList # 리스트 반환