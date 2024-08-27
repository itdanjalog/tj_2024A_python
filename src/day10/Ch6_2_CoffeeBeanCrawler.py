from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import datetime

from selenium import webdriver
import time

#[CODE 1]
def CoffeeBean_store(result):
    CoffeeBean_URL = "https://www.coffeebeankorea.com/store/store.asp"
    wd = webdriver.Chrome() #교재 코드 수정
             
    for i in range(1, 389):  #마지막 매장번호(최근 신규 매장번호) 까지 반복 # 389
        wd.get(CoffeeBean_URL)
        try:
            wd.execute_script( f'storePop2({i})' )
            time.sleep(1) #스크립트 실행 할 동안 1초 대기
            html = wd.page_source
            soupCB = BeautifulSoup(html, 'html.parser')
            store_name_h2 = soupCB.select("div.store_txt > h2")
            store_name = store_name_h2[0].string
            print(store_name)  #매장 이름 출력하기
            store_info = soupCB.select("div.store_txt > table.store_table > tbody > tr > td")
            store_address_list = list(store_info[2])
            store_address = store_address_list[0]
            store_phone = store_info[3].string
            store = [store_name,store_address,store_phone]
            result.append( store )
            print( store )
        except:
            continue 
    return

#[CODE 0]
def main():
    result = []
    print('CoffeeBean store crawling >>>>>>>>>>>>>>>>>>>>>>>>>>')
    CoffeeBean_store(result)  #[CODE 1]
    
    CB_tbl = pd.DataFrame(result, columns=('store', 'address','phone'))
    CB_tbl.to_csv('CoffeeBean.csv', encoding='utf-8', mode='w', index=True ,  index_label=['index'])


if __name__ == '__main__':
     main()
