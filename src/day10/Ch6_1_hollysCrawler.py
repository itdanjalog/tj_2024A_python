from bs4 import BeautifulSoup
import urllib.request
import pandas as pd

#[CODE 1]
def hollys_store(result):
    for page in range(1,59):
        Hollys_url = f'https://www.hollys.co.kr/store/korea/korStore.do?pageNo={page}'
        print(Hollys_url)
        html = urllib.request.urlopen(Hollys_url)
        soupHollys = BeautifulSoup(html, 'html.parser')
        tag_tbody = soupHollys.find('tbody')
        for store in tag_tbody.find_all('tr'):
            if len(store) <= 3:
                break
            store_td = store.find_all('td')
            store_name = store_td[1].string
            store_sido = store_td[0].string
            store_address = store_td[3].string
            store_phone = store_td[5].string
            result.append([store_name]+[store_sido]+[store_address]
                          +[store_phone])
        print( result )
    return

#[CODE 0]
def main():
    result = []
    print('Hollys store crawling >>>>>>>>>>>>>>>>>>>>>>>>>>')
    hollys_store(result)   #[CODE 1] 호출 
    print( result )
    #hollys_tbl = pd.DataFrame( result, columns=( 'store', 'sido-gu', 'address','phone'))
    #hollys_tbl.to_csv('hollys.csv', encoding='cp949', mode='w', index=True ,  index_label=['index'] )

if __name__ == '__main__':
     main()
