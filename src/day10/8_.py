from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

driver = webdriver.Chrome()
url = "https://www.starbucks.co.kr/store/store_map.do?disp=locale"
driver.get(url)
time.sleep(1)

# "서울" 클릭
driver.find_element (By.CSS_SELECTOR, ".loca_search > h3 > a").click()
time.sleep(5)

# "전체" 클릭
driver.find_element ( By.CSS_SELECTOR,  ".sido_arae_box > li > a").click()
time.sleep(5)

driver.find_element ( By.CSS_SELECTOR,  ".gugun_arae_box > li > a").click()
time.sleep(10)

html = driver.page_source
soup = BeautifulSoup(html,"html.parser")

for i in range( 0 , 616 ) :

    store = soup.select("li.quickResultLstCon")[i]

    name = store.select("strong")[0].text.strip()
    lat = store["data-lat"]
    lng = store["data-long"]
    type = store.select("i")[0]["class"][0][4:]
    address = str(store.select("p.result_details")[0]).split('<br/>')[0].split('>')[1]
    tel = str(store.select("p.result_details")[0]).split('<br/>')[1].split('</p>')[0]

    print( name )