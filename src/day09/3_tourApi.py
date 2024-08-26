# day09 > 3_tourApi.py
# 준비
# (1)공공데이터포털 : https://www.data.go.kr -> 로그인
# (2)출입국관광통계서비스 검색
# (3)활용신청
import json
import urllib.request

# 인증키
인증키 = "nwPZ%2F9Z3sVtcxGNXxOZfOXwnivybRXYmyoIDyvU%2BVDssxywHNMU2tA55Xa8zvHWK0bninVkiuZAA4550BDqIbQ%3D%3D"

# [code2]
def getRequestUrl( url ) :
    요청객체 = urllib.request.Request( url )
    try :
        응답객체 = urllib.request.urlopen( 요청객체 )
        if 응답객체.getcode() == 200 :
            return 응답객체.read().decode('utf-8')
    except Exception as e :
        return None

# [code3] 지정한 날짜 , 지정한 국가 , 구분 을 받아서 url 요청 하기
def getTourismStateItem( yyyymm , nat_cd , ed_cd ) :
    # 1. 출입국관광통계 의 기본 URL
    base = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
    # 2. 매개변수 : 인증키 , 연월 , 국가코드 , 출입국구분코드
    parameters  = f'?_type=json&serviceKey={ 인증키 }'
    parameters += f'&YM={ yyyymm }&NAT_CD={ nat_cd }&ED_CD={ed_cd}'
    url = base + parameters # 3. 합치기
    print( f'>> url : {url}')
    # http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList?serviceKey=인증키&YM=연도월&NAT_CD=국가코드&ED_CD=출입국구분

    responseDecode = getRequestUrl( url ) # 4. url 요청후 응답객체 받기
    if responseDecode == None : return None # 5. 만약에 응답객체가 None 이면 None 반환
    else: return json.loads( responseDecode ) # 6. 만약에 응답객체가 존재하면 json형식을 파이썬 객체로 반환 함수
        # PY :  json.loads( ) JSON형식 --> PY형식 변환 함수  , json.dumps( ) PY형식 ---> JSON형식(문자열타입) 변환 함수
        # JS : JSON.parse( ) JSON형식 --> JS형식 변환 함수  , JSON.stringify( ) JS형식 ---> JSON형식(문자열타입) 변환 함수
# [ code4 ]
def getTourismStateService( nat_cd , ed_cd ,  nStartYear , nEndYear ) :
    # jsonResult = [] #  수집한 데이터를 저장할 리스트 객체
    # dataEND = f'{ str(nEndYear)}{str(12)}' # 마지막 연도의 마지막 월 (12)
    isDataEnd = 0 # 데이터의 끝 확인하는 변수

    for year in range( nStartYear , nEndYear+1  ) : # 시작연도 부터 마지막연도 까지 반복 # range( 부터 , 미만 )
        print( f'>>year : { year }')
        for month in range( 1 , 13 ) : # 1 ~ 13(미만) , 1~12월까지 반복
            print( f'>>month : { month }')
            if isDataEnd == 1 : break # 만약에 isDataEnd가 1이면 반복문 종료

            # :0>2  오른쪽정렬 > , 2 자릿수 , 0 빈칸이면 0 채움 #   1 -> 01 , 10 -> 10 , 5 -> 05
            yyyymm = f'{ str(year)}{ str(month):0>2}'
            print( yyyymm )

            # 지정한 날짜 , 지정한 국가 , 구분 을 전달하여 요청후 응답
            jsonData = getTourismStateItem( yyyymm , nat_cd , ed_cd )
            if jsonData != None :
                print( jsonData )

# [code1]
def main() :
    # API 서비스 요청시 필요한 매개변수들을 입력받기
    nat_cd = int( input("국가코드(중국:112 / 일본:130 / 미국:275) : "))# 2. 국가코드 ( 공식 문서 참고 )
    nStartYear = int( input('시작연도 : '))# 3. 데이터 수집 시작 연도
    nEndYear = int( input('끝연도 : ')) # 4. 마지막 데이터의 연도
    ed_cd = 'E' # 5. 구분( E: 방한외래관광객(입국) , D : 해외 출국 )

    # 6. 서비스 요청후 응답 객체 받기
    jsonResult = []  # 1. 수집한 데이터를 저장할 리스트 객체
    jsonResult = getTourismStateService( nat_cd , ed_cd , nStartYear , nEndYear )
    print( jsonResult ) # 확인

if __name__ == "__main__" :
    main()

