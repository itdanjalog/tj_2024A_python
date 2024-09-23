import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. 데이터 생성 (주제: 과일, 추가 특성 포함)
data = {
    'weight': [110, 160, 130, 320, 370, 300, 55, 65, 60, 210, 220, 200, 90, 80, 100, 190, 180, 170, 100, 90,
               140, 280, 320, 130, 200, 140, 250, 150, 70, 80, 200, 300, 220, 140, 180, 230, 220, 250],
    'sweetness': [6.2, 7.2, 6.8, 8.1, 8.6, 8.1, 5.2, 5.7, 6.1, 7.2, 7.6, 6.7, 7.3, 6.9, 7.3, 7.5, 7.4, 7.3, 7.0, 6.8,
                  6.9, 8.0, 8.1, 6.7, 7.0, 6.6, 7.8, 7.1, 6.7, 6.5, 7.0, 7.6, 7.3, 7.0, 7.2, 7.5, 7.4, 7.7]
}
df = pd.DataFrame(data)

# 2. KMeans 군집 분석
kmeans = KMeans( n_clusters= 3 )
kmeans.fit( df )
print(kmeans.cluster_centers_) # 이 중심들은 각각의 군집에 속한 과일들의 평균적인 무게와 당도를 나타냅니다.

# 3. 군집 결과를 데이터프레임에 추가
df['cluster'] = kmeans.labels_

print( df )

# 4. 새로운 데이터 포인트 (새로운 과일)
newData = {
    'weight': [110],
    'sweetness': [7]
}
newDf = pd.DataFrame( newData )

# 5. 새로운 데이터의 군집 예측
new_clusters = kmeans.predict(newDf)
print( new_clusters )

# 7. 군집 시각화 (무게와 단맛을 기준으로 시각화)
plt.scatter(df['weight'], df['sweetness'], c=df['cluster'], marker='o', label='old data')
plt.scatter(newDf['weight'], newDf['sweetness'], c='red', marker='^', label='new data')
plt.xlabel('weight (g)')
plt.ylabel('sweetness (1-10)')
plt.title('KMeans')
plt.colorbar(label='군집')
plt.legend()
plt.show()
















'''
1. 특성의 스케일 차이 (Scaling Issue)
무게는 110~370g의 범위로 변화합니다.
반면, 당도는 5.2~8.6 정도의 작은 범위에서 변화합니다.
이처럼 무게의 범위가 훨씬 크기 때문에 군집 분석에서 무게가 더 큰 영향을 미치게 됩니다. KMeans는 유클리드 거리를 기준으로 군집화하는데, 스케일이 큰 특성(무게)이 더 많이 반영되어 군집의 기준이 무게 중심으로 나타날 수 있습니다.
'''
# 이 변환을 통해 머신러닝 알고리즘에서 성능이 더 좋아지거나 학습이 더 빠르게 진행될 수 있습니다.

# 스케일
from sklearn.preprocessing import StandardScaler
# 데이터 표준화 (무게와 당도의 스케일 맞추기)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['weight', 'sweetness']])

# KMeans 군집 분석
kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_data)
print(kmeans.cluster_centers_) # 이 중심들은 각각의 군집에 속한 과일들의 평균적인 무게와 당도를 나타냅니다.
# 결과 출력
df['cluster'] = kmeans.labels_
print(df)

newData = {
    'weight': [110],
    'sweetness': [7]
}
newDf = pd.DataFrame( newData )
scaled_new_data  = scaler.fit_transform(newDf[['weight', 'sweetness']])

new_clusters = kmeans.predict( scaled_new_data  )
print( new_clusters )

plt.scatter( scaled_data[ : , 0], scaled_data[ : , 1], c=df['cluster'], marker='o', label='old data')
plt.scatter( scaled_new_data[ : , 0], scaled_new_data[ : , 1], c='red', marker='^', label='new data')
plt.xlabel('weight (g)')
plt.ylabel('sweetness (1-10)')
plt.title('KMeans')
plt.colorbar(label='군집')
plt.legend()
plt.show()



import matplotlib.pyplot as plt

# 가상의 SSE 데이터 예시 # 이 경우, 3이 최적의 클러스터 수로 판단될 수 있습니다.
cluster_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sse = [400, 200, 100, 80, 75, 70, 68, 67, 66, 65]  # 예시 데이터

plt.plot(cluster_count, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method Example')
plt.xticks(cluster_count)
plt.grid()
plt.show()


# 최적의 클러스터 수 ( 엘보우 )
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sse = []
cluster_range = range(1, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)  # SSE

plt.plot(cluster_range, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()
'''
엘보우 방법은 클러스터 수에 따라 KMeans의 **총 제곱 오차(SSE)**를 계산하고, 그 결과를 시각화하여 최적의 클러스터 수를 찾는 방법입니다. SSE는 군집 내의 데이터 포인트와 클러스터 중심 간의 거리를 제곱하여 합한 값입니다.

단계:
다양한 클러스터 수 (예: 1부터 10까지)에 대해 KMeans를 수행합니다.
각 클러스터 수에 대해 SSE를 계산합니다.
SSE를 클러스터 수에 대해 그래프로 그립니다.
그래프에서 SSE의 변화가 급격히 줄어드는 지점을 찾습니다. 이 지점이 최적의 클러스터 수입니다.
'''


kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_data)
print(kmeans.cluster_centers_) # 이 중심들은 각각의 군집에 속한 과일들의 평균적인 무게와 당도를 나타냅니다.
# 결과 출력
df['cluster'] = kmeans.labels_
print(df)

newData = {
    'weight': [110],
    'sweetness': [7]
}
newDf = pd.DataFrame( newData )
scaled_new_data  = scaler.fit_transform(newDf[['weight', 'sweetness']])

new_clusters = kmeans.predict( scaled_new_data  )
print( new_clusters )

plt.scatter( scaled_data[ : , 0], scaled_data[ : , 1], c=df['cluster'], marker='o', label='old data')
plt.scatter( scaled_new_data[ : , 0], scaled_new_data[ : , 1], c='red', marker='^', label='new data')
plt.xlabel('weight (g)')
plt.ylabel('sweetness (1-10)')
plt.title('KMeans')
plt.colorbar(label='군집')
plt.legend()
plt.show()