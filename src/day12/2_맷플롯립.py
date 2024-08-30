# day12 > 2_맷플롯립.py
# - 파이썬 에서 데이터를 시각화 해주는 패키지
# - 데이터 분석결과를 시각화 하여 직관적으로 이해하기 위해서 사용
# - 선(라인) 차트 , 바 차트 , 파이(원형) 차트 , 히스토그램 , 산점도 등등 지원하는 라이브러리

# [1] 설치
'''
방법1 : import matplotlib 에 커서를 두고 빨간 느낌표 클릭후 -> install matplotlib
방법2 : 상단메뉴 -> 파일 -> 설정 -> 왼쪽메뉴[프로젝트] 하위 [인터프리터]
        -> [+] alt+insert -> matplotlib 검색 후 패키지 선택 -> [패키지 설치]
'''
# [2] 모듈 가져오기
import matplotlib
# [3] 버전 확인
print( matplotlib.__version__ ) # 3.9.2
# [4] pyplot 모듈 가져오기
import matplotlib.pyplot as plt

# [5] 라인 플롯 차트
# 1. 차트에 표시할 데이터 샘플 데이터 준비
x = [ 2016,2017,2018,2019,2020 ]
y = [ 350,410,520,695,543 ]
# 2. 라인플롯( 선 차트)에 x축과 y축 지정하여 라인플롯 생성 , .plot(  )
plt.plot( x , y )
# 3. 차트 제목 지정       , .title( )
plt.title('Annual Sales')
# 4. x축 레이블(축제목) 설정 , .xlabel( )
plt.xlabel('year')
# 5. y축 레이블(축제목) 설정 , .ylabel( )
plt.ylabel('sales')
# 6.차트 실행               , .show( )
plt.show()

# [6] 바 플롯 차트
# 1. 데이터 준비
y1 = [ 350,410,520,695]
y2 = [200,250,385,350]
x = range( len(y1) ) # 0부터 y1 리스트의 길이만큼  리스트 생성 [ 0,1,2,3 ]
# 2. x축과 y축 데이터를 지정하여 바 차트 생성 , # .bar( ) #width : 막대굵기 #color 막대색상
plt.bar( x , y1 , width=0.7 , color = 'blue' )
plt.bar( x , y2 , width=0.7 , color = 'red'  , bottom=y1 )
# 3. 차트 제목
plt.title('Quarterly sales')
# 4. x축 레이블
plt.xlabel( 'Quarters')
# 5. y축 레이블
plt.ylabel( 'sales' )
# 6. 눈금 이름 리스트 생성
xLabel = [ 'first' , 'second' , 'third' , 'fourth' ]
plt.xticks( x , xLabel , fontsize = 10 )
# 7. 범례( 막대구분 이름 표시 )
plt.legend( [ 'chairs' , 'desks' ] )
# 8. 차트 실행
plt.show()

# [7] 산점도 : x축 과 y축의 값 관계를 시각화 #각 데이터 포인트는 두 변수의 값을 x축y축에 대응시켜 점으로 표현
# 1. 데이터 준비
import random
x = [ random.random() for _ in range( 50 ) ] # 50개의 요소를 난수(0~1)로 생성 하여 리스트 생성
print( x )
y = [ random.random() for _ in range( 50 ) ]
print( y )
# 2. 산점도 차트 생성
plt.scatter( x , y )
#
plt.show( )
















































