### 도미 데이터 준비하기
# 도미(bream) 데이터 준비: 도미의 길이와 무게 리스트로 각각 저장
# bream_length는 도미의 길이를 저장한 리스트이고,
# bream_weight는 도미의 무게를 저장한 리스트입니다.

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

import matplotlib.pyplot as plt

# plt.scatter(bream_length, bream_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

### 빙어 데이터 준비하기
# 빙어(smelt) 데이터 준비: 빙어의 길이와 무게 리스트로 각각 저장
# smelt_length는 빙어의 길이를 저장한 리스트이고,
# smelt_weight는 빙어의 무게를 저장한 리스트입니다.

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

## 첫 번째 머신러닝 프로그램
# 도미와 빙어의 데이터를 합쳐서 전체 물고기의 길이와 무게 데이터를 준비
# 길이 리스트와 무게 리스트를 각각 병합하여 하나의 리스트로 만듭니다.
length = bream_length+smelt_length
weight = bream_weight+smelt_weight
# 물고기의 길이와 무게를 하나의 리스트로 묶은 2차원 리스트를 생성
# 각 물고기의 길이와 무게를 하나의 [길이, 무게] 형태의 리스트로 묶어서 fish_data라는 리스트에 저장합니다.
fish_data = [[l, w] for l, w in zip(length, weight)]
print(fish_data)

# 물고기의 정답 데이터 준비: 도미는 1, 빙어는 0으로 레이블링
# 도미 35마리와 빙어 14마리의 정답 레이블을 만들어 fish_target 리스트에 저장
fish_target = [1]*35 + [0]*14 # 도미 35마리 빙어14마리
print(fish_target)
# K-최근접 이웃 알고리즘을 사용하기 위해 sklearn의 KNeighborsClassifier를 임포트
from sklearn.neighbors import KNeighborsClassifier
# KNeighborsClassifier 객체를 생성하여 kn 변수에 저장
kn = KNeighborsClassifier() #(n_neighbors의 기본값은 5)
# 모델을 학습시킴: fish_data(입력 데이터)와 fish_target(정답 데이터)을 사용하여 kn 객체에 모델 학습을 시킴
kn.fit(fish_data, fish_target)
# 모델의 정확도 평가: 학습 데이터에 대한 정확도를 계산하여 반환
kn.score(fish_data, fish_target) # 정확도

### k-최근접 이웃 알고리즘
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.scatter(30, 600, marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 새로운 데이터 [30, 600] (길이 30cm, 무게 600g인 물고기)을 사용하여 분류 예측
# 이 값이 도미(1)인지 빙어(0)인지 예측
print( kn.predict([[30, 600]]) )
# print(kn._fit_X)
# print(kn._y)

# K-최근접 이웃 알고리즘에서 이웃의 수를 49로 설정하여 새로운 모델 kn49 생성
kn49 = KNeighborsClassifier(n_neighbors=49)
# kn49 모델을 동일한 데이터로 학습시킴
kn49.fit(fish_data, fish_target)
# 새로운 kn49 모델의 정확도 평가: 49개의 이웃을 기준으로 학습 데이터에 대한 정확도 계산
kn49.score(fish_data, fish_target)
# 도미가 35마리, 이웃의 수는 49이므로 이웃 중에서 도미의 비율을 계산
print(35/49)
