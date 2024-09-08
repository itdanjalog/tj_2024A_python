# # 11장. 회귀분석 (1) 결정트리분석


# 2) 데이터 준비 및 탐색

import numpy as np
import pandas as pd

print( pd.__version__ )

# 피처 이름 파일 읽어오기
feature_name_df = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/features.txt', sep='\s+',  header=None, names=['index', 'feature_name'], engine='python')

print( feature_name_df.head())

print( feature_name_df.shape )

# index 제거하고, feature_name만 리스트로 저장
feature_name = feature_name_df.iloc[:, 1].values.tolist()

print( feature_name[:5]  )

X_train = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/train/X_train.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_train.columns = feature_name

X_test = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/test/X_test.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_test.columns = feature_name

Y_train = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/train/y_train.txt', sep='\s+', header = None,names = ['action'], engine = 'python')
Y_test = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/test/y_test.txt' , sep = '\s+', header = None,names = ['action'], engine = 'python')

print( X_train.shape, Y_train.shape, X_test.shape, Y_test.shape )

print( X_train.info() )

print( X_train.head() )

print(Y_train['action'].value_counts())


label_name_df = pd.read_csv('./UCI_HAR_Dataset/UCI_HAR_Dataset/activity_labels.txt', sep='\s+',  header=None, names=['index', 'label'], engine='python')

# index 제거하고, feature_name만 리스트로 저장
label_name = label_name_df.iloc[:, 1].values.tolist()

print( label_name )

# 3) 모델 구축 : 결정트리모델

from sklearn.tree import DecisionTreeClassifier
# 결정 트리 분류 분석 : 1) 모델 생성
dt_HAR = DecisionTreeClassifier(random_state=156)

# 결정 트리 분류 분석 : 2) 모델 훈련
dt_HAR.fit(X_train, Y_train)

# 결정 트리 분류 분석 : 3) 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = dt_HAR.predict(X_test)

# 4) 결과 분석
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)
print('결정 트리 예측 정확도 : {0:.4f}'.format(accuracy))

# ** 성능 개선을 위해 최적 파라미터 값 찾기

print('결정 트리의 현재 하이퍼 파라미터 : \n', dt_HAR.get_params())

# 최적 파라미터 찾기 - 1
from sklearn.model_selection import GridSearchCV
params = {   'max_depth' : [ 6, 8, 10, 12, 16, 20, 24] }

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',    cv=5, return_train_score=True)
grid_cv.fit(X_train , Y_train)

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
print(cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']])

print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))

# 최적 파라미터 찾기 - 2
params = {
    'max_depth' : [ 8, 16, 20 ],
    'min_samples_split' : [ 8, 16, 24 ]
}

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',
                       cv=5, return_train_score=True)
grid_cv.fit(X_train , Y_train)

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
print( cv_results_df[['param_max_depth','param_min_samples_split', 'mean_test_score', 'mean_train_score']] )

print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))

best_dt_HAR = grid_cv.best_estimator_
best_Y_predict = best_dt_HAR.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)

print('best 결정 트리 예측 정확도 : {0:.4f}'.format(best_accuracy))


# ** 중요 피처 확인하기
import seaborn as sns
import matplotlib.pyplot as plt

feature_importance_values = best_dt_HAR.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)

feature_top10 = feature_importance_values_s.sort_values(ascending=False)[:10]

plt.figure(figsize = (10, 5))
plt.title('Feature Top 10')
sns.barplot(x=feature_top10, y=feature_top10.index)
plt.show()

# 5) Graphviz를 사용한 결정트리 시각화
    # 1) graphviz 패키지 다운로드 및 설치
    # - https://graphviz.org/download/에서 graphviz-2.49.3 (64-bit) EXE installer[sha256]을 다운로드하여 설치하기
    # (3) 파이썬 래퍼 모듈 graphviz를 pip 명령으로 Anaconda에 설치


from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성.
export_graphviz(best_dt_HAR, out_file="./tree.dot", class_names=label_name , feature_names = feature_name,
                impurity=True, filled=True)

import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화
with open("./tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)


















