# 12장. 군집분석 : 타깃마케팅을 위한 K-평균 군집화

import pandas as pd
import math
# https://archive.ics.uci.edu/dataset/352/online+retail 에서 다운로드


# ### ====================================== [1] 'Online_Retail.xlsx' 파일에 저장된 데이터를 retail_df라는 데이터프레임에 저장합니다. =================================
retail_df = pd.read_excel('Online_Retail.xlsx')
retail_df = retail_df[retail_df['Quantity'] > 0]    # # ### [2] 이를 통해 유효한 거래 데이터만 남기고, 오류나 비정상적인 데이터를 제거합니다. # 오류 데이터 정제 상품 수량이 0보다 크고
retail_df = retail_df[retail_df['UnitPrice'] > 0]   # 단가가 0보다 크며
retail_df = retail_df[retail_df['CustomerID'].notnull()] # 고객 ID가 결측되지 않은
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int) # ### [3] 'CustomerID' 자료형을 정수형으로 변환
'''
고객 ID는 일반적으로 소수점이 없는 정수형 데이터이기 때문에, 데이터 분석 시 고객 ID를 적절한 형태로 맞추기 위해서 실수형이나 문자열 데이터를 정수형으로 변환합니다.
이렇게 하면, 이후 분석 및 그룹화 작업에서 정수형 고객 ID로 효율적으로 처리할 수 있습니다.
참고: 이 코드를 사용하기 전에는 CustomerID 열에 결측값이 없어야 하며, 모든 값이 정수형으로 변환 가능한 형식이어야 오류가 발생하지 않습니다.
'''
retail_df.drop_duplicates(inplace=True) # ### [4] drop_duplicates() drop_duplicates()는 pandas의 메서드로, 중복된 행을 제거하는 기능을 수행합니다.  #중복 레코드 제거
'''
# ### [5]  - 제품 수, 거래건 수, 고객 수 탐색
고유 값 개수를 각각 상품 코드, 거래 번호, 고객 ID에 대해 계산하고, 
이 값을 데이터프레임으로 생성하여 각 열의 이름을 지정합니다. 
결과 데이터프레임은 단일 행으로 구성되며, 각 열에는 해당하는 고유 값의 개수가 저장됩니다.
'''
# pd.DataFrame([{'Product':len(retail_df['StockCode'].value_counts()),
#               'Transaction':len(retail_df['InvoiceNo'].value_counts()),
#               'Customer':len(retail_df['CustomerID'].value_counts())}],
#              columns = ['Product', 'Transaction', 'Customer'],
#             index = ['counts'])
# print( retail_df['Country'].value_counts() )
retail_df['SaleAmount'] = retail_df['UnitPrice'] * retail_df['Quantity'] # ### [6] # 주문금액 컬럼 추가

# ### [7] - 고객의 마지막 주문후 경과일(Elapsed Days), 주문횟수(Freq), 주문 총액(Total Amount) 구하기
aggregations = {
    'InvoiceNo':'count', # 열의 값을 세어 각 고객의 거래 수를 계산합니다.
    'SaleAmount':'sum', #  열의 합계를 구해 각 고객의 총 판매 금액을 계산합니다.
    'InvoiceDate':'max' # 열에서 각 고객의 가장 최신 거래 날짜를 구합니다.
}
'''
retail_df를 'CustomerID'로 그룹화하여 각 고객별로 집계 작업을 수행합니다.
groupby('CustomerID')는 고객 ID별로 데이터를 그룹화합니다.
agg(aggregations)는 위에서 정의한 집계 작업을 각 그룹에 적용하여 결과를 계산합니다.
'''
customer_df = retail_df.groupby('CustomerID').agg(aggregations)
'''
reset_index() 메소드는 customer_df의 인덱스를 재설정합니다. 
그룹화 후 인덱스가 고객 ID로 설정되어 있는데, 이를 기본 정수 인덱스로 변경합니다.
이 과정은 인덱스가 데이터프레임의 열로 변환되어, 고객 ID가 일반 열로 추가되며,
 데이터프레임의 구조가 더 직관적으로 됩니다.
'''
customer_df = customer_df.reset_index()
#print(  customer_df.head() ) #작업 확인용 출력
customer_df = customer_df.rename(columns = {'InvoiceNo':'Freq', 'InvoiceDate':'ElapsedDays'}) # ### [8]  컬럼이름 바꾸기 #  열 이름을 변경하여 'InvoiceNo'을 'Freq'로, 'InvoiceDate'을 'ElapsedDays'로 변경합니다
#print( customer_df.head() ) #작업 확인용 출력

# ### [9]   - 마지막 구매후 경과일 계산하기
import datetime
customer_df['ElapsedDays'] = datetime.datetime(2011,12,10) - customer_df['ElapsedDays'] # customer_df['ElapsedDays'] 열의 각 날짜에서 이 기준 날짜를 빼서 경과된 일수를 계산합니다
#print( customer_df.head() ) #작업 확인용 출력
customer_df['ElapsedDays'] = customer_df['ElapsedDays'].apply(lambda x: x.days+1)
'''
apply() 메소드를 사용하여 'ElapsedDays' 열의 각 timedelta 객체에 대해 경과 일수만 추출합니다.
lambda x: x.days + 1 함수는 timedelta 객체의 .days 속성으로부터 일수를 추출하고, 여기에 1을 추가합니다.
일수에 1을 추가하는 이유는 일반적으로 경과된 날짜를 계산할 때, 현재 날짜를 포함시키기 위해 1일을 추가할 수 있습니다. (이는 상황에 따라 다를 수 있습니다.)
'''
#print( customer_df.head() ) #작업 확인용 출력


#### [10] - 현재 데이터 값의 분포 확인하기 # 하기 전
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# customer_df['Freq']: 각 고객의 거래 수
# customer_df['SaleAmount']: 각 고객의 총 판매 금액
# customer_df['ElapsedDays']: 각 고객의 경과 일수
ax.boxplot([customer_df['Freq'], customer_df['SaleAmount'], customer_df['ElapsedDays']], sym='bo')
plt.xticks([1, 2, 3], ['Freq', 'SaleAmount','ElapsedDays' ])
plt.show()

#### [11] - 데이터 값의 왜곡(치우침)을 줄이기 위한 작업 : 로그 함수로 분포 조정
import numpy as np
# np.log1p() 함수는 입력값에 1을 더한 후 로그를 계산합니다. log1p(x)는 log(1 + x)와 같습니다.
# 'Freq' 열의 값에 1을 더하고 로그를 취해 새로운 열 'Freq_log'에 저장합니다. 로그 변환은 데이터의 분포를 더 정규 분포에 가깝게 조정하는 데 유용합니다.
customer_df['Freq_log'] = np.log1p(customer_df['Freq'])
customer_df['SaleAmount_log'] = np.log1p(customer_df['SaleAmount'])
customer_df['ElapsedDays_log'] = np.log1p(customer_df['ElapsedDays'])
customer_df.head()  #작업 확인용 출력
# 조정된 데이터 분포를 다시 박스플롯으로 확인하기
fig, ax = plt.subplots()
ax.boxplot([customer_df['Freq_log'], customer_df['SaleAmount_log'],customer_df['ElapsedDays_log']], sym='bo')
plt.xticks([1, 2, 3], ['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'])
plt.show()

# *log1p 테스트 샘플
# 원래 데이터
data = np.array([1, 10, 100, 1000]) # 원래 데이터: [1, 10, 100, 1000] # 원래 데이터는 1, 10, 100, 1000처럼 큰 차이를 보여
# 로그 변환
log_data = np.log1p(data)
print("원래 데이터:", data)
print("로그 변환된 데이터:", log_data) # 변환된 데이터는 [0, 2.3, 4.6, 6.9]처럼 큰 차이를 줄여주고, 숫자들의 차이가 더 균형 잡혀 보이게 돼요.
# 로그 변환은 데이터의 큰 값들을 줄이고 작은 값들을 더 눈에 띄게 만들어서, 데이터의 분포가 더 균형 잡히게 해줘요.
# 이렇게 변환하면, 데이터가 더 정규 분포에 가까워지기도 해요. 정규 분포는 대개 데이터를 중심으로 고르게 분포시키는 모양이에요.


#### [12] 3) 모델 구축 : K-평균 군집화 모델
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
X_features = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']].values
# 정규 분포로 다시 스케일링하기
from sklearn.preprocessing import StandardScaler
print( X_features )
X_features_scaled = StandardScaler().fit_transform(X_features)
print( X_features_scaled )
# StandardScaler를 사용하여 데이터를 정규화합니다.
# fit_transform() 메소드는 두 가지 작업을 수행합니다:
# fit(): X_features의 평균과 표준편차를 계산합니다.
# transform(): 이 평균과 표준편차를 사용하여 데이터를 정규화합니다.
# 결과적으로 X_features_scaled에는 평균이 0이고 표준편차가 1인 데이터가 저장됩니다.

# 로그 변환은 데이터의 분포를 더 정규화하는 데 사용되며, 큰 값의 영향을 줄입니다.
# 스케일링은 데이터의 범위를 표준화하여 모든 변수의 중요성을 동일하게 만들어줍니다.
# 데이터 전처리에서 이 두 단계를 모두 수행하는 이유는 로그 변환으로 데이터의 분포를 조정하고, 스케일링으로 각 변수의 영향을 균등하게 하여 KMeans와 같은 클러스터링 알고리즘의 성능을 높이기 위해서입니다.

#### [13] KMeans 모델 학습
'''
KMeans 클러스터링을 사용하여 데이터를 5개의 클러스터로 나누고, 각 데이터 포인트에 대해 어떤 클러스터에 속하는지 레이블을 붙입니다. 
결과적으로, customer_df 데이터프레임에는 각 고객이 속한 클러스터를 나타내는 'Cluster' 열이 추가됩니다.
'''
kmeans = KMeans(n_clusters=5, random_state=0)  # n_clusters는 군집의 개수입니다
# customer_df['Cluster'] = kmeans.fit_predict(X_features_scaled)
kmeans.fit(X_features)
customer_df['Cluster'] = kmeans.labels_

import matplotlib.pyplot as plt

# 2D 시각화
plt.figure(figsize=(10, 7))
plt.scatter(customer_df['Freq'], customer_df['SaleAmount'], c=customer_df['Cluster'],  marker='o', label='old data' )
plt.title('KMeans Clustering - 2D Visualization')
plt.xlabel('Frequency (log)')
plt.ylabel('Sale Amount (log)')
plt.colorbar(label='군집')
plt.legend(title='Cluster')
plt.show()


kmeans = KMeans(n_clusters=5, random_state=0)  # n_clusters는 군집의 개수입니다
# customer_df['Cluster'] = kmeans.fit_predict(X_features_scaled)
kmeans.fit(X_features_scaled)
customer_df['Cluster'] = kmeans.labels_
# 2D 시각화
plt.figure(figsize=(10, 7))
plt.scatter(customer_df['Freq_log'], customer_df['SaleAmount_log'], c=customer_df['Cluster'], marker='o', label='old data')
plt.title('KMeans Clustering - 2D Visualization')
plt.xlabel('Frequency (log)')
plt.ylabel('Sale Amount (log)')
plt.colorbar(label='군집')
plt.legend(title='Cluster')
plt.show()



#### [14] - 최적의 k 찾기 (1) 엘보우 방법
distortions = [] # 디스톨션
# 클러스터링에서의 왜곡: KMeans와 같은 클러스터링 기법에서는 클러스터 중심과 데이터 포인트 간의 거리의 제곱합을 의미합니다. 이 값이 클수록 데이터 포인트가 클러스터 중심에서 멀리 떨어져 있다는 것을 나타내며, 클러스터링이 덜 잘 이루어졌다는 의미입니다.
# Distortion은 왜곡이라는 뜻을 가지며, 데이터 분석에서는 클러스터 중심과 데이터 포인트 간의 거리의 제곱합을 의미합니다.

# 왜곡도를 저장할 리스트를 생성합니다. 왜곡도는 각 클러스터의 중심과 데이터 포인트 간의 거리의 제곱합입니다.
for i in range(1, 20):
# 클러스터 개수를 1부터 10까지 변화시키면서 반복합니다. 이 범위에서 최적의 클러스터 개수를 찾으려고 합니다.
    kmeans_i = KMeans(n_clusters=i, random_state=0, n_init='auto')  # 모델 생성
    # 현재 클러스터 개수(i)로 KMeans 모델을 생성합니다. # random_state=0은 재현성을 보장합니다.
    # n_init='auto'는 KMeans의 초기화 반복 횟수를 자동으로 설정합니다. (버전 1.4부터는 기본값이 10입니다.)
    kmeans_i.fit(X_features_scaled)  # 모델 훈련
    distortions.append(kmeans_i.inertia_)
    # kmeans_i.inertia_는 왜곡도(distortion)를 계산하여 저장합니다. 이 값은 클러스터 중심과 데이터 포인트 간의 거리 제곱합입니다.

plt.plot(range(1, 20), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


'''
그래프에서 왜곡도가 급격히 감소하다가 감소폭이 줄어드는 지점을 찾아 클러스터 개수를 결정합니다. 
이 지점을 "팔꿈치(elbow)"라고 부르며, 이 점이 적절한 클러스터 개수를 나타냅니다.
'''
'''
왜곡도는 데이터가 얼마나 잘 클러스터(군집)되어 있는지를 나타내는 수치입니다
    왜곡도는 클러스터의 중심과 데이터 포인트 간의 거리의 제곱합입니다
[잘 나눈 상자]:
공들이 상자 안에서 서로 가까이 모여 있는 상태
왜곡도가 낮음
[잘못 나눈 상자]:
공들이 상자 안에서 멀리 떨어져 있는 상태
왜곡도가 높음

클러스터의 중심과 데이터 포인트 간의 거리를 측정하여, 데이터가 얼마나 잘 그룹화되었는지를 나타냅니다. 클러스터가 잘 형성되면 왜곡도가 낮고, 그렇지 않으면 높습니다.

'''


kmeans = KMeans(n_clusters=6, random_state=0, n_init='auto') # 모델 생성

# 모델 학습과 결과 예측(클러스터 레이블 생성)
Y_labels = kmeans.fit_predict(X_features_scaled)

customer_df['ClusterLabel'] = Y_labels

print( customer_df.head() )  #작업 확인용 출력


# 샘플
import matplotlib.cm as cm
kmeans = KMeans(n_clusters=6, random_state=0, n_init='auto')
Y_labels = kmeans.fit_predict(X_features_scaled)

# 클러스터 색상 설정 및 데이터 시각화
for i in range(6):

    # 현재 클러스터 인덱스 i에 대해 색상을 설정합니다. cm.jet 함수는 연속적인 색상 값을 생성하며, float(i) / 6은 0부터 1까지의 값으로 변환하여 색상 맵에서 색상을 선택합니다.
    plt.scatter(X_features_scaled[Y_labels == i, 0], X_features_scaled[Y_labels == i, 1],
                marker='o', edgecolor='black', s=50, label='cluster ' + str(i))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



















































"""
## 4) 결과 분석 및 시각화
### - 최적의 k 찾기 (2) 실루엣 계수에 따른 각 클러스터의 비중 시각화 함수 정의
from matplotlib import cm

def silhouetteViz(n_cluster, X_features):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init='auto')
    Y_labels = kmeans.fit_predict(X_features)

    silhouette_values = silhouette_samples(X_features, Y_labels, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster):
        c_silhouettes = silhouette_values[Y_labels == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(float(c) / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)

    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.title('Number of Cluster : ' + str(n_cluster) + '\n' \
              + 'Silhouette Score : ' + str(round(silhouette_avg, 3)))
    plt.yticks(y_ticks, range(n_cluster))
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()

clusterScatter(6, X_features_scaled)
def clusterScatter(n_cluster, X_features):
    c_colors = []
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init='auto')
    Y_labels = kmeans.fit_predict(X_features)

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster)  # 클러스터의 색상 설정
        c_colors.append(c_color)
        # 클러스터의 데이터 분포를 동그라미로 시각화
        plt.scatter(X_features[Y_labels == i, 0], X_features[Y_labels == i, 1],
                    marker='o', color=c_color, edgecolor='black', s=50,
                    label='cluster ' + str(i))

        # 각 클러스터의 중심점을 삼각형으로 표시
    for i in range(n_cluster):
        plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1],
                    marker='^', color=c_colors[i], edgecolor='w', s=200)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

silhouetteViz(3, X_features_scaled) #클러스터 3개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(4, X_features_scaled) #클러스터 4개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(5, X_features_scaled) #클러스터 5개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(6, X_features_scaled) #클러스터 6개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
clusterScatter(3, X_features_scaled) #클러스터 3개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(4, X_features_scaled)  #클러스터 4개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(5, X_features_scaled)  #클러스터 5개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(6, X_features_scaled)  #클러스터 6개인 경우의 클러스터 데이터 분포 시각화

### 결정된 k를 적용하여 최적의 K-mans 모델 완성
best_cluster = 4

kmeans = KMeans(n_clusters=best_cluster, random_state=0, n_init='auto')
Y_labels = kmeans.fit_predict(X_features_scaled)

customer_df['ClusterLabel'] = Y_labels

customer_df.head()   #작업 확인용 출력

#### - ClusterLabel이 추가된 데이터를 파일로 저장
customer_df.to_csv('Online_Retail_Customer_Cluster.csv')

## << 클러스터 분석하기 >>
### 1) 각 클러스터의 고객수
customer_df.groupby('ClusterLabel')['CustomerID'].count()
### 2) 각 클러스터의 특징
customer_cluster_df = customer_df.drop(['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'],axis=1, inplace=False)

# 주문 1회당 평균 구매금액 : SaleAmountAvg
customer_cluster_df['SaleAmountAvg'] = customer_cluster_df['SaleAmount']/customer_cluster_df['Freq']

customer_cluster_df.head()

# 클러스터별 분석
customer_cluster_df.drop(['CustomerID'],axis=1, inplace=False).groupby('ClusterLabel').mean()
"""

