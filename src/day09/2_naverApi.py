# 크롤링 > 2_naverApi.py
#
import json
import urllib.request

# 개발자 개발자센터에서 애플리케이션 신청후 발급받은 키 와 비밀버호
네이버키 = 'jUlR3SZSDOLH433RVxcD'
네이버비밀번호 = 'Nlgyt3NDOs'

#[ code2  ]지정한 URL의 요청을 실행하고 응답을 받는 함수
def getRequestUrl( url ):
    요청객체 = urllib.request.Request( url )                        # 2. 지정한 URL 설정
    요청객체.add_header("X-Naver-Client-Id" , 네이버키 )            # 3. HTTP요청 객체내 HEADER 정보 추가
    요청객체.add_header('X-Naver-client-Secret', 네이버비밀번호)      # 4. http요청시 네이버api id와 비밀번호 전송

    try: #예외처리
        응답객체 = urllib.request.urlopen( 요청객체)            # 5. 지정한 url 실행후 응답 반환
        print( f'>>code2 요청URL 결과 상태 : { 응답객체.getcode() }') # 확인
        if 응답객체.getcode() == 200 :                          # 6. 만약에 응답의 상태가 2xx 이면 성공
            return 응답객체.read().decode('utf-8')                # 7. 실행된 URL내 모든 내용물 읽어오기
    except Exception as e :
        return  None                                         #8. 없으면 None

#[ code3  ] 매개변수로 검색대상 , 검색어 , 시작번호 , 한번에표기할개수 를 받아서
# URL 구성하여 getRequestUrl() 메소드에게 요청하여 응답객체를 받아서 JSON형식 으로 반환 함수
def getNaverSearch( node , srcText , page_start , display ) :
    # URL 파라미터 공문 : https://developers.naver.com/docs/serviceapi/search/news/news.md#파라미터
    base  = "https://openapi.naver.com/v1/search"# 1. 요청url의 기본주고
    node = f'/{node}.json' # 2. 요청url의 검색 대상 과 json 파일이름
        # https://openapi.naver.com/v1/search/news.json
    parameters = f'?query={ urllib.parse.quote( srcText ) }&start={ page_start }&display={ display }' # 3. 요청url의 파라미터
        # https://openapi.naver.com/v1/search/news.json?query=검색어&start=시작번호&display=한번표기할개수
    url = base + node + parameters # 4.url 합치기
    print( f'>>code3 요청URL : { url }')  # 확인 # https://openapi.naver.com/v1/search/news.json?query=월드컵&start=1&display=100
    responseDecode = getRequestUrl( url )# 5. url 요청을 하고 응답 객체 받기 , [code2]
    if responseDecode == None : return None # 6. 만약에 url 응답 객체가 없으면 None 반환
    else : return json.loads( responseDecode )  # 7. 응답객체가 있으면 JSON 형식으로 변환
        # json.loads( 문자열 ) : JSON 형식 으로 변환 함수

# [ CODE 4 ]
def getPostData( post , jsonResult , cnt ) :
    # 응답받은 객체의 요소들 공문 : https://developers.naver.com/docs/serviceapi/search/news/news.md#응답
    title = post['title'] # 뉴스 기사의 제목
    description = post['description'] # 뉴스 기사의 내용을 요약한 패시지 정보
    org_link = post['originallink'] # 뉴스 기사 원문의 URL
    link = post['link'] # 뉴스 기사의 네이버 뉴스 URL

    dic = { 'cnt' : cnt , 'title' : title , 'description' : description , 'org_link' : org_link , 'link' : link } # 딕셔너리 생성
    jsonResult.append( dic ) # 리스트에 딕셔너리 저장

# [code 1]
def main() :
    node = 'news' # 1. 크롤링할 대상 [ 네이버 제공하는 검색대상 : 1.news 2.blog 3.shop 등등 ] - 공문 참고 #https://developers.naver.com/docs/serviceapi/search/search.md
    srcText = input('검색어 입력하세요:') # 2. 사용자 입력으로 받은 검색어 변수
    cnt = 0 # 3. 검색 결과 개수
    jsonResult = [] # 4. 검색 결과를 정리하여 저장할 리스트 변수

    # 5. 1부터 100까지의 검색 결과를 처리한다. # [code2]네이버 뉴스 검색 결과에 대한 응답을 저장하는 객체
    jsonResponse = getNaverSearch( node , srcText , 1 , 100 ) # 5. [code 3]
        # jsonResponse{ total:총 검색 결과 개수 , start:검색 시작 위치 , display:한 번에 표시할 검색 결과 개수 , item : 개별 검색 결과  }
        # JSON 형식의 결괏값에서는 items 속성의 JSON 배열로 개별 검색 결과를 반환합니다.
        # URL : # https://openapi.naver.com/v1/search/news.json?query=월드컵&start=1&display=100
    print( f'>> jsonResponse : { jsonResponse }')

    total = jsonResponse['total']# 6. 전체 검색 결과 개수

    # 7. 응답객체가 None 이 아니면서 응답객체의 display 가 0이 아니면 무한반복 , url 응답객체가 없을때 까지
    while ( ( jsonResponse != None ) and (jsonResponse['display'] != 0 ) ) :
        # 8. 검색결과리스트(items) 에서 하나씩 item(post) 호출 # 공문 : https://developers.naver.com/docs/serviceapi/search/news/news.md#뉴스
        for post in jsonResponse['items'] : # 응답받은 검색 결과 중에서 한 개를 저장한객체
            cnt+=1 # 응답 개수 변수 1증가
            # 9. [code 3] 검색 결과 한개를 처리한다.
            getPostData( post , jsonResult , cnt )
        # 10. start 를 display 만큼 증가 시킨다
        start = jsonResponse['start'] + jsonResponse['display']
        # 11. 첫번째 요청 1 , 100  , 두번째 요청 101 , 100 , 세번째 요청 201 , 100
        # 무료버전 기준으로 : start : 1001 오류가 발생하면서 종료된다. 1001이상 하기 위해서는 API 유료 계약 해야한다.
        jsonResponse = getNaverSearch( node , srcText , start , 100 )
    #
    print( f'전체 검색 : { total }건')
    print( f'가져온 데이터(무료기준) : { cnt }건')
    # print( jsonResult ) # 확인

    # (방법1) PY 객체를 JSON형식(문자열) 변환후 파일 처리 # with open() as 파일변수명 : # with 종료시 자동으로 파일닫기
    with open( f'{srcText}-naver-{node}.json' , 'w' , encoding='utf-8' ) as file :
        jsonFile = json.dumps( jsonResult , indent=4 , sort_keys=True , ensure_ascii=False )
        file.write( jsonFile ) # 파일 쓰기
'''
    # (방법2)  PY 객체를 JSON형식(문자열) 변환후 파일 처리
    file = open( f'{srcText}-naver-{node}.json' , 'w' , encoding='utf-8' ) # 파일 쓰기 모드 객체 생성 # # 월드컵-naver-news.json
        # 2. json.dumps() : py객체를 JSON형식의 문자열로 반환 함수
            # json.dumps( 변환할PY객체 , indent=들여쓰기수준 , sort_keys=알바벳으로정렬 , ensure_ascii=아스키문자 )
            # (1) 변환할py객체 : 딕셔너리 또는 리스트 # jsonResult : 검색 결과를 정리하여 저장할 리스트 변수
            # (2) indent : 생략시 들여쓰기없음 , 주로 4 정도가 가독성이 좋다.
            # (3) sort_keys : True( key값 기준으로 알파벳순 정렬 ) , False( 딕셔너리 키 순서대로 )
            # (4) ensure_ascii : False( UTF-8 인코딩으로 비아스키코드 문자 - 주로 한글 ) True( 아스키코드 문자 )
    jsonFile = json.dumps( jsonResult , indent=4 , sort_keys=True , ensure_ascii=False )
    file.write( jsonFile ) # 파일 쓰기
    file.close() # 파일 닫기
'''


if __name__ == "__main__" :
    main() # [code1] 메소드 실행









