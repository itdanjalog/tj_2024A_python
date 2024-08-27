from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request

# wd = webdriver.Chrome() #교재 코드 수정
#
# wd.get("https://finance.naver.com/")
#
# html = wd.page_source

#print( html )
response = urllib.request.urlopen('https://finance.naver.com/' )

soupCB = BeautifulSoup( response.read() ,'html.parser' , from_encoding='euc-kr')
#soupCB = BeautifulSoup( html, 'html.parser')
store_name_h2 = soupCB.select("#_topItems1 > .up")
# print(store_name_h2)

for tr in store_name_h2 :
    th = tr.select('th > a')
    print( th[0].text )
    td = tr.select('td')

    print( td[0].text )
    print( td[1].text)
    print( td[2].text.strip() )


'''
주요 차이점:
**string**은 텍스트 한 조각만 포함하고 내부에 다른 요소가 없는 요소를 처리할 때 유용합니다. 자식이 여러 명 있으면 'None'을 반환합니다.
**text**는 텍스트 노드가 얼마나 깊이 중첩되어 있는지에 관계없이 요소 내의 모든 텍스트를 검색하므로 더 유연하고 강력합니다.
요약하면:
중첩되지 않은 단일 텍스트 문자열이 필요한 경우 **string**을 사용하세요.
중첩된 요소의 내용을 포함하여 모든 텍스트 내용을 검색하려면 **text**를 사용하세요.
'''