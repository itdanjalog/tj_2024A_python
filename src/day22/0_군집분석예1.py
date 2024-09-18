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
kmeans = KMeans( n_clusters=5 )
kmeans.fit( df )

# 3. 군집 결과를 데이터프레임에 추가
df['cluster'] = kmeans.labels_

print( df )

# 4. 새로운 데이터 포인트 (새로운 과일)
new_data = pd.DataFrame({
    'weight': [110],
    'sweetness': [7]
})

# 5. 새로운 데이터의 군집 예측
new_clusters = kmeans.predict(new_data)

# 6. 결과 출력
for i, point in new_data.iterrows():
    print(f"새로운 과일 (무게: {point['weight']}, 단맛: {point['sweetness']} 은 군집 {new_clusters[i]}에 속합니다.")

# 7. 군집 시각화 (무게와 단맛을 기준으로 시각화)
plt.figure(figsize=(12, 8))
plt.scatter(df['weight'], df['sweetness'], c=df['cluster'], cmap='viridis', marker='o', label='old data')
plt.scatter(new_data['weight'], new_data['sweetness'], c='red', marker='^', label='new data')
plt.xlabel('weight (g)')
plt.ylabel('sweetness (1-10)')
plt.title('KMeans')
plt.colorbar(label='군집')
plt.legend()
plt.show()
