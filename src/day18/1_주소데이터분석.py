# 9장. 지리 정보 분석 (1) 주소데이터분석+맵
# 1. 데이터 수집
import pandas as pd
CB = pd.read_csv('CoffeeBean.csv', encoding='CP949', index_col=0, header=0, engine='python')
print( CB.head() )  #작업 내용 확인용 출력

# ======================================== ======================================== #
# 2. 데이터 준비 및 탐색
# 시/도 행정구역 이름 정규화
addr = []

for address in CB.address:
    addr.append(str(address).split())

#작업 내용 확인용 출력
print('데이터 개수 : %d' % len(addr))
print( addr )

# ======================================== ======================================== #
# - addr에서 행정구역 표준 이름이 아닌것 수정하기
addr2 = []

# addr에서 행정구역 표준 이름이 아닌것 수정하기
for i in range(len(addr)):
    if addr[i][0] == "서울": addr[i][0]="서울특별시"
    elif addr[i][0] == "서울시": addr[i][0]="서울특별시"
    elif addr[i][0] == "부산시": addr[i][0]="부산광역시"
    elif addr[i][0] == "인천": addr[i][0]="인천광역시"
    elif addr[i][0] == "광주": addr[i][0]="광주광역시"
    elif addr[i][0] == "대전시": addr[i][0]="대전광역시"
    elif addr[i][0] == "울산시": addr[i][0]="울산광역시"
    elif addr[i][0] == "세종시": addr[i][0]="세종특별자치시"
    elif addr[i][0] == "경기": addr[i][0]="경기도"
    elif addr[i][0] == "충북": addr[i][0]="충청북도"
    elif addr[i][0] == "충남": addr[i][0]="충청남도"
    elif addr[i][0] == "전북": addr[i][0]="전라북도"
    elif addr[i][0] == "전남": addr[i][0]="전라남도"
    elif addr[i][0] == "경북": addr[i][0]="경상북도"
    elif addr[i][0] == "경남": addr[i][0]="경상남도"
    elif addr[i][0] == "제주": addr[i][0]="제주특별자치도"
    elif addr[i][0] == "제주도": addr[i][0]="제주특별자치도"
    elif addr[i][0] == "제주시": addr[i][0]="제주특별자치도"

    addr2.append(' '.join(addr[i]))

print( addr2 )

# ======================================== ======================================== #


addr2 = pd.DataFrame(addr2, columns=['address2'])

print( addr2 )

# ======================================== ======================================== #

CB2 = pd.concat([CB, addr2],  axis=1 )

print(CB2.head() ) #작업 내용 확인용 출력

# ======================================== ======================================== #
# - 지도 정보 시각화 라이브러리 설치 및 임포트하기
CB2.to_csv('CoffeeBean_2.csv',encoding='CP949', index = False)
# ======================================== ======================================== #

# 1) 숭례문 좌표를 사용하여 지도 객체 테스트하기
import folium
map_osm = folium.Map(location=[37.560284, 126.975334], zoom_start = 16)
map_osm.save('map.html')

# ======================================== ======================================== #
# 2) 정리해둔 CoffeeBean_2.csv 파일 로드
CB_file = pd.read_csv('CoffeeBean_2.csv',encoding='cp949',  engine='python')
print( CB_file.head() ) #작업 내용 확인용 출력

# ======================================== ======================================== #

# 3) 지오서비스웹(GEOSERVICE-WEB)을 사용하여 구한 GPS 좌표 파일 로드
'''
CB_geoData = pd.read_csv('CB_geo_sph_2.csv',encoding='utf8',  engine='python')
print( len(CB_geoData) ) #작업 내용 확인용 출력

map_CB = folium.Map(location=[37.560284, 126.975334], zoom_start = 15)
for i, store in CB_geoData.iterrows():
    folium.Marker(location=[store['_Y'], store['_X']], popup= store['store'], icon=folium.Icon(color='red', icon='star')).add_to(map_CB)

map_CB.save('map_CB.html')
'''

import webbrowser
webbrowser.open('map_CB.html')







