# 실습1 : 삼성전자 의 최근 1년 시세 정보 CSV 로 다운로드 받아서 판다스(데이터프레임) 으로 읽어오기
# 정보데이터시스템 : http://data.krx.co.kr/
        # 1. 데이터프레임 객체를 콘솔 에 출력 ( CSV -> 데이터프레임 )
        # 2. 삼성전자 의 최근 1년 시세 중 일자(X)별 종가(Y)를 막대차트로 표현하시오.
# 1. 데이터 자료 준비 : data_5209_20240829.csv
# 2. csv 파일을 판다스의 데이터프레임 가져오기
import pandas as pd
try : pd = pd.read_csv( 'data_5209_20240829.csv'   ) # utf-8 기본값 으로 인코딩
except Exception as e : # utf-8 인코딩 오류이면 cp949 인코딩하기
    pd = pd.read_csv( 'data_5209_20240829.csv' , encoding='cp949'  )
print( pd )
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 0: invalid start byte
# 주의할점 : 파일들의 인코딩 방식 ( utf-8 , cp949 , ISO-8859 등등 )
# 3. 데이터프레임 의 특정 열 호출 # 데이터프레임['열이름']
print( pd['일자'] ) # 일자 열 만 호출
print( pd['종가'] ) # 종가 열 만 호출

# 4. 시각화 준비
import matplotlib.pyplot as plt
# x = [ '2024/08/29' , '2024/08/30' , '2024/08/31' ]
# y = [ '123123123' , '456456456' , '321321321' ]
x = pd['일자']    # 판다스 데이터프레임의 일자(열) 를 x축
y = pd['종가']    # 판다스 데이터프레임의 종가(열) 를 y축
plt.plot( x , y )
plt.title('chart')
plt.xlabel('date')
plt.ylabel('price')
plt.show( )





