# day24 > 1_군집분석.py
# 비지도 학습 : 타깃(종속)값이 주어지지 않은 상태에서 학습 수행  vs  지도 학습: 회귀분석,분류분석
# 군집분석 : 데이터 포인트(군집)를 사전에 정의된 군집개수(K)로 그룹화하여 유사한 데이터들을 군집에 배치 하여 새로운 데이터의 군집 예측
    # 특징 : 1.비지도학습 2.사전에 클러스터수(K) 3. K-평균알고리즘
# 군집/클러스터 화 : 학습을 수행하여 데이터 간의 관계를 분석하고 이에 따른 유사한 데이터들을 군집으로 구성하는 작업
# k-평균 알고리즘 : K개의 클러스터를 구성하는 알고리즘
# 최적의 클러스터 수(K) 찾기 : 1.엘보방법 2.실루엣방법

# [1] 데이터 수집 # 종속변수 없음
    # weight : 과일 무게 # sweetness : 과일 당도
data = {
    'weight': [110, 160, 130, 320, 370, 300, 55, 65, 60, 210, 220, 200, 90, 80, 100, 190, 180, 170, 100, 90,
               140, 280, 320, 130, 200, 140, 250, 150, 70, 80, 200, 300, 220, 140, 180, 230, 220, 250],
    'sweetness': [6.2, 7.2, 6.8, 8.1, 8.6, 8.1, 5.2, 5.7, 6.1, 7.2, 7.6, 6.7, 7.3, 6.9, 7.3, 7.5, 7.4, 7.3, 7.0, 6.8,
                  6.9, 8.0, 8.1, 6.7, 7.0, 6.6, 7.8, 7.1, 6.7, 6.5, 7.0, 7.6, 7.3, 7.0, 7.2, 7.5, 7.4, 7.7]
}
import pandas as pd
df = pd.DataFrame( data )
# [2] k-평균 군집 분석 모델 # KMeans
from sklearn.cluster import KMeans
model = KMeans( n_clusters = 3 )        # 군집분석 모델 객체 생성 # n_clusters = 클러스터수
model.fit( df )                         # 모델에 데이터를 피팅 # 비지도 학습 이므르 종속변수가 없다.
# [3] 클러스터/군집/중심지 확인 # .cluster_centers_
print( model.cluster_centers_)
''' 3개 리스트 : 3개의 군집 : 각 군집들의 평균 무게 와 당도 표시 # [0] : 무게 , [1] 당도
[[315.           8.08333333]  [101.76470588   6.63529412] [205.33333333   7.32      ]] '''
# [4] 군집 결과를 확인 # .labels_
print( model.labels_ ) # 0~2 :총3개 # [1 2 1 0 0 0 1 1 1 2 2 2 1 1 1 2 2 2 1 1 1 0 0 1 2 1 2 1 1 1 2 0 2 1 2 2 2 2]
# [5] 결과를 데이터프레임에 추가
df['cluster'] = model.labels_
print( df ) # 무게와 당도에 따른 군집(번호)를 확인
# [6] 새로운 데이터로 군집 예측
newData = { 'weight' : [110] , 'sweetness' : [7] } # 새로운 과일 의 무게 와 당도 1개
newDf = pd.DataFrame( newData )
# [7] 예측
Cpred = model.predict( newDf )
print( Cpred )

# 시각화
import matplotlib.pyplot as plt
plt.scatter( df['weight'] , df['sweetness'] , c = df['cluster'] , marker = 'o' ) # 산점도( x와 y 관계 )
plt.scatter( newDf['weight'] , newDf['sweetness'] , marker='^' )
plt.show()

# 개선1 : 무게와 당도의 범위의 차이가 크다. 스케일 차이가 크다. # 무게는 100단위 , 당도 1 단위
# - 데이터 분석 에서 스케일 차이가 크면 특정 속성에 비중를 많이 차지 된다. #
# - 스케일 표준화 : 알고리즘에서 성능을 개선하여 좀 더 좋은 학습 하기위해서 필요한 작업 # StandardScaler
# [1] 스케일 객체 생성
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
#    무게와 당도의 스케일 맞추기 # 데이터 표준화 # 무게 100단위 , 당도 1단위 # 특정 알고리즘(k평균)에서 무게에 더 큰 비중 갖는다.
    # 여러개 데이터들의 평균을 0 으로 하고 표준편차를 1이 되도록 변환 하는 과정 : 표준화
    # 표준편차 크다 : 데이터가 평균으로 부터의 차이 # 평균으로 부터 멀리 떨어져 있다. # 데이터가 매우 다양하다.  # [ 1,100,500,1000 ]
    # 표준편차 작다 : 데이터가 평균으로 부터의 차이 # 평균으로 부터 가깝게 있다. # 데이터가 비슷비슷하다        # [ 2,4,6,8,10]
scaledData = scaler.fit_transform( df[['weight' , 'sweetness'] ] )
print( data )           # 스케일 전 # [ 110  ,  6.2 ]
print( scaledData )     # 스케일 후 # [ -0.81219782 -1.40395189]
# [2] 스케일 된 데이터로 모델 학습
model2 = KMeans( n_clusters= 3 )
model2.fit( scaledData ) # 스케일 된 데이터를 학습
df['cluster'] = model2.labels_ # 클러스터 결과를 데이터프레임에 대입
df['weight_scale'] = scaledData[  :  , 0 ] # 모든 행 의 첫번째 열 추출 ( 스케일 된 무게 )
df['sweetness_scale'] = scaledData[ : , 1 ] # 모든 행의 두번째 열 추출 ( 스케일 된 당도 )
# [3] 새로운 데이터 예측
scaledNewData = scaler.fit_transform( newDf[ [ 'weight' , 'sweetness'] ]) # 새로운 데이터 스케일
Cpred2 = model2.predict( scaledNewData )
# 시각화
plt.scatter( df['weight_scale'] , df['sweetness_scale'] , c = df['cluster'] , marker = 'o' )
plt.scatter( scaledNewData[ :  , 0 ] , scaledNewData[ :  , 1 ] , marker='^' )
plt.show()

# 개선2 : 최적의 k(클러스터 수) 를 찾기 # 엘보방법 # .inertia_
    # 그래프에서 SSE(왜곡)의 변화가 급격히 줄여드는 지점을 찾는다. 그 지점이 최적의 클러스터 수
sse = [ ] # 오차들을 저장하는 리스트
for 클러스터수 in range( 1 , 11 ) : # 1 ~ 10 까지의 클러스터 수를 테스트 # 10회전
        model = KMeans( n_clusters= 클러스터수 )
        model.fit( scaledData )
        print( model.inertia_ ) # 데이터들의 거리 차이의 제곱값 합을 오차/왜곡(SSE) 를 계산하고 반환한다.
        sse.append( model.inertia_ ) # SSE를 리스트에 대입
# 총 클러스터 1개 일때 부터 10개 까지의 모델 10개 SSE(오차)를 리스트에 저장
print( sse )
# 오차 시각화
plt.plot( sse , marker = 'o' ) # 선 차트
plt.show()

# 최적의 K 확인후 재 모델링
model3 = KMeans( n_clusters = 2 )
model3.fit( scaledData )

# 클러스터 3이 최적의 수(K)
클러스터수 = [ 1 , 2 , 3 , 4 , 5 ]
왜곡 = [ 400 , 200 , 100 , 80 , 75 ]  # 급격하게 변화가 줄여드는 지점 ( 클러스터수가 3에서 4로 변화될때 )
          # 200  #100  #20   #5
              #50%  #80%  #75%   변화 비율 = 가장큰 100->80 # 급격하게 변화했다.
















