# 11장 분류분석 (2) 결정 트리 분석을 이용한 사용자 움직임 분류 프로젝트

# #### - https://archive.ics.uci.edu/에 접속하여, ‘human activity recognition’를 검색한다.
#  -'human+activity+recognition+using+smartphones.zip’을 다운받아서, 압축을 풀고, 폴더를 [11장_data] 폴더로 이동한다.
# https://archive.ics.uci.edu/ # Human Activity Recognition Using Smartphones
from sklearn import tree
import matplotlib.pyplot as plt  # 맷플롯립 에서 폰트 관련 객체

import pandas as pd

# 피처 이름 파일 읽어오기
'''
# [1] 피처 이름 파일 읽어오기
'features.txt' 파일을 읽어서 feature_name_df라는 DataFrame에 저장합니다.
파일의 각 열은 공백(\s+)으로 구분되어 있으며, 파일에는 헤더가 없기 때문에 header=None으로 설정합니다.
열 이름을 'index'와 'feature_name'으로 설정합니다.
engine='python'은 Pandas의 내부 파서 대신 Python의 기본 파서를 사용하도록 하는 설정입니다.
'''
feature_name_df = pd.read_csv('features.txt', sep='\s+',  header=None, names=['index', 'feature_name'], engine='python')



'''
# [2] 파일 구조 확인
feature_name_df의 상위 5개 행을 출력하고, 데이터 프레임의 형태 (행과 열의 개수)를 확인합니다.
'''
# print( feature_name_df.head() )
# print( feature_name_df.shape )



# index 제거하고, feature_name만 리스트로 저장
'''
[3] 피처 이름만 리스트로 저장
iloc을 사용해 두 번째 열인 feature_name만을 선택하고, 이를 리스트로 변환하여 feature_name이라는 변수에 저장합니다.
이 리스트에는 데이터셋의 각 열에 해당하는 이름들이 저장됩니다.
'''
feature_name = feature_name_df.iloc[:, 1].values.tolist()
# print( feature_name[:5] )



'''
[4]X_train과 X_test 데이터 읽기
'X_train.txt' 및 'X_test.txt' 파일을 읽어옵니다. 이 파일들은 공백으로 구분된 값들로 되어 있으며, 
데이터에 헤더가 없으므로 header=None으로 지정합니다.
encoding='latin-1'은 파일을 특정 인코딩으로 읽기 위한 설정입니다.
X_train.columns = feature_name은 앞서 얻은 feature_name 리스트를 각 열의 이름으로 설정합니다.
'''
X_train = pd.read_csv('X_train.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_train.columns = feature_name
X_test = pd.read_csv('X_test.txt', delim_whitespace=True, header=None, encoding='latin-1')
X_test.columns = feature_name

'''
[5] Y_train과 Y_test 레이블 읽기
'y_train.txt'와 'y_test.txt' 파일을 읽어옵니다. 이 파일들 역시 공백(\s+)으로 구분되어 있으며, 
헤더가 없고, 읽은 데이터의 열 이름을 'action'으로 지정합니다.
'''
Y_train = pd.read_csv('y_train.txt', sep='\s+', header = None,names = ['action'], engine = 'python')
Y_test = pd.read_csv('y_test.txt' , sep = '\s+', header = None,names = ['action'], engine = 'python')



'''
[6] 6. 데이터 구조 및 정보 확인
X_train, Y_train, X_test, Y_test의 크기를 출력하여 데이터셋의 형태를 확인합니다.
X_train.info()로 데이터의 요약 정보를 확인합니다. 이는 각 열의 데이터 타입과 누락된 값 여부 등을 보여줍니다.
X_train.head()는 X_train의 상위 5개 행을 출력합니다.
Y_train['action'].value_counts()는 Y_train의 'action' 열에서 각 값(레이블)의 빈도를 출력합니다. 이를 통해 레이블 분포를 확인할 수 있습니다.
'''
# print( X_train.shape, Y_train.shape, X_test.shape, Y_test.shape )
# print(X_train.info() )
# print( X_train.head() )
# print(Y_train['action'].value_counts())



'''
[7] 7. 레이블 이름 파일 읽기 # 각 타켓의 번호가 무엇인지 정의한 파일인듯.
'activity_labels.txt' 파일을 읽어와 label_name_df라는 데이터 프레임에 저장합니다.
두 번째 열에 해당하는 'label' 열만 추출하여 리스트로 변환한 후 label_name 변수에 저장합니다. 
이 리스트에는 레이블에 대한 이름들이 저장됩니다.
'''
label_name_df = pd.read_csv('activity_labels.txt', sep='\s+',  header=None, names=['index', 'label'], engine='python')
# index 제거하고, feature_name만 리스트로 저장
label_name = label_name_df.iloc[:, 1].values.tolist()
# print( label_name )



# ### 3) 모델 구축 : 결정트리모델
'''
[1] 1. 결정 트리 모델 임포트
DecisionTreeClassifier는 Scikit-learn 라이브러리에서 제공하는 결정 트리 모델입니다.
이 모델은 분류 문제에 사용되며, 데이터를 분할해가면서 의사결정 규칙을 만들어서 분류 작업을 수행합니다.
'''
from sklearn.tree import DecisionTreeClassifier




# 결정 트리 분류 분석 : 1) 모델 생성
'''
[2] 2. 결정 트리 모델 생성
DecisionTreeClassifier()를 사용해 결정 트리 분류 모델 객체를 생성하고, dt_HAR 변수에 저장합니다.
random_state=156는 의사결정 규칙을 만들 때 무작위 요소가 있는 경우 이를 고정하여 결과가 재현 가능하게 설정한 것입니다. 
즉, 같은 데이터와 같은 설정을 사용할 때 항상 동일한 트리가 생성되도록 보장합니다.
'''
dt_HAR = DecisionTreeClassifier(random_state=156 , max_depth=6)




'''
[3]3. 모델 훈련 (학습)
fit() 메소드를 사용하여 결정 트리 모델을 훈련시킵니다.
X_train은 입력 데이터(특성 데이터)이고, Y_train은 각 입력 데이터에 해당하는 레이블(정답 데이터)입니다.
모델은 이 데이터를 바탕으로 트리를 구성하며, 훈련이 끝나면 학습된 결정 트리가 생성됩니다.
'''
# 결정 트리 분류 분석 : 2) 모델 훈련
dt_HAR.fit(X_train, Y_train)



''''
[4] 4. 평가 데이터에 대한 예측 수행
predict() 메소드를 사용하여 테스트 데이터(X_test)에 대한 예측을 수행합니다.
이 과정에서 모델은 학습된 결정 트리를 사용하여 X_test의 각 샘플이 어떤 레이블에 속하는지 예측하고, 그 결과를 Y_predict 변수에 저장합니다.
Y_predict는 테스트 데이터에 대한 예측 결과로, 예측된 레이블이 들어있는 배열입니다.
'''
# 결정 트리 분류 분석 : 3) 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = dt_HAR.predict(X_test)
# print( Y_predict )

"""

"""
### 4) 결과 분석
'''
[1] 1. accuracy_score 함수 임포트
accuracy_score는 Scikit-learn 라이브러리에서 제공하는 함수로, 모델이 예측한 값과 실제 값이 얼마나 일치하는지를 계산해주는 함수입니다.
정확도(accuracy)-"애-큐-러-시" 는 전체 샘플 중에서 모델이 올바르게 예측한 샘플의 비율을 나타냅니다.
'''
from sklearn.metrics import accuracy_score


'''
[2]정확도 계산
accuracy_score() 함수에 실제 레이블(Y_test)과 예측된 레이블(Y_predict)을 전달하여 정확도를 계산합니다.
Y_test는 모델이 예측하려고 했던 실제 정답(레이블)이고, Y_predict는 결정 트리 모델이 예측한 결과입니다.
두 배열의 값이 얼마나 일치하는지를 기반으로 정확도가 계산되며, 결과는 accuracy 변수에 저장됩니다.
'''
accuracy = accuracy_score(Y_test, Y_predict)
print('결정 트리 예측 정확도 : {0:.4f}'.format(accuracy))


# 모델의 결정 트리 구조를 출력 (선택 사항)


tree.plot_tree(dt_HAR, feature_names=feature_name, class_names=label_name, filled=True)
plt.show()


"""
#### ** 성능 개선을 위해 최적 파라미터 값 찾기
'''
[1]1. 결정 트리 하이퍼파라미터 출력 , ?????
get_params() 메소드는 결정 트리 모델(dt_HAR)에 설정된 현재 하이퍼파라미터를 출력합니다.
결정 트리 모델에는 여러 가지 하이퍼파라미터가 존재하며, 이 코드에서는 그 값들을 출력하여 확인할 수 있습니다.
'''
#print('결정 트리의 현재 하이퍼 파라미터 : \n', dt_HAR.get_params())



'''
[2]
GridSearchCV는 다양한 하이퍼파라미터 조합을 시도해 최적의 하이퍼파라미터를 찾는 데 사용되는 도구입니다.
교차 검증(Cross-validation)을 기반으로 각 하이퍼파라미터 조합의 성능을 평가하고, 가장 성능이 좋은 하이퍼파라미터를 선택합니다.
'''
from sklearn.model_selection import GridSearchCV




'''
[3] 하이퍼파라미터 후보 설정
params 변수에는 결정 트리 모델의 max_depth 하이퍼파라미터에 대한 후보값들이 정의되어 있습니다.
max_depth는 트리의 최대 깊이를 나타내며, 값이 클수록 트리가 깊어져 더 복잡한 모델이 됩니다.
여기서는 6, 8, 10, 12, 16, 20, 24의 값들이 시도됩니다.
'''
params = { 'max_depth' : [ 6, 8, 10, 12, 16, 20, 24] }




'''
[4] 4. GridSearchCV 객체 생성
GridSearchCV 객체를 생성합니다.
dt_HAR는 튜닝할 모델, param_grid=params는 하이퍼파라미터의 범위(이 경우 max_depth의 값들),
scoring='accuracy'는 모델 평가를 정확도로 할 것임을 의미합니다.
cv=5는 5-폴드 교차 검증을 의미합니다. 데이터셋을 5개로 나누어, 5번 반복해서 모델을 학습 및 평가합니다.
return_train_score=True는 각 교차 검증에 대해 훈련 데이터에 대한 점수도 함께 반환한다는 의미입니다.
'''
grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',
                       cv=5, return_train_score=True)




'''
[5]교차 검증을 통한 모델 학습
fit() 메소드를 호출해 GridSearchCV를 실행합니다.
교차 검증을 통해 다양한 max_depth 값으로 모델을 학습한 후, 최적의 하이퍼파라미터를 찾습니다.
'''
grid_cv.fit(X_train , Y_train)



'''
[6]교차 검증 결과 출력
grid_cv.cv_results_는 교차 검증 결과를 담고 있는 딕셔너리로, 이를 데이터프레임으로 변환합니다.
각 max_depth 값에 대한 교차 검증 결과 중, param_max_depth, mean_test_score, mean_train_score 열을 선택해 출력합니다.
    param_max_depth: 사용된 max_depth 값.
    mean_test_score: 교차 검증에서의 테스트 데이터 평균 정확도.
    mean_train_score: 교차 검증에서의 훈련 데이터 평균 정확도.
'''
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
# print( cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']] )


'''
[7] 7. 최적 하이퍼파라미터와 최고 정확도 출력
grid_cv.best_score_는 교차 검증 중 최고 정확도를 기록한 값을 의미합니다.
grid_cv.best_params_는 최고의 성능을 낸 하이퍼파라미터(이 경우 max_depth)의 값을 반환합니다.
'''
print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))





"""
#### 최적 파라미터 찾기 - 2
from sklearn.model_selection import GridSearchCV
'''
[1] 1. 하이퍼파라미터 범위 설정

이 부분에서 결정 트리의 두 가지 하이퍼파라미터인 max_depth(트리의 최대 깊이)와 min_samples_split(노드를 분할하기 위한 최소 샘플 수)에 대해 여러 값들을 정의하고 있습니다.
max_depth: 트리의 최대 깊이로, 8, 16, 20을 각각 시도합니다.
min_samples_split: 노드를 분할하기 위해 필요한 최소 샘플 수로, 8, 16, 24의 값을 시도합니다.
이 값들을 조합하여 다양한 결정 트리 모델을 테스트할 수 있습니다.
'''
params = {
    'max_depth' : [ 8, 16, 20 ],
    'min_samples_split' : [ 8, 16, 24 ]
}



'''
[2] 2. GridSearchCV 객체 생성
GridSearchCV 객체를 생성합니다. 여기서:
    dt_HAR: 우리가 튜닝하려는 결정 트리 모델.
    param_grid=params: 하이퍼파라미터 조합으로 설정된 params 값을 전달합니다.
    scoring='accuracy': 모델을 평가할 때 정확도를 기준으로 합니다.
    cv=5: 5-폴드 교차 검증을 수행합니다.
    return_train_score=True: 교차 검증 시 훈련 데이터에 대한 점수도 반환합니다.
'''

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy',
                       cv=5, return_train_score=True)



'''
[3]3. 하이퍼파라미터 조합을 통해 모델 학습
fit() 메소드를 사용하여 X_train과 Y_train 데이터를 기반으로 모델을 학습합니다.
GridSearchCV는 정의된 max_depth와 min_samples_split의 다양한 조합을 시도하면서 최적의 하이퍼파라미터를 찾아냅니다.
'''
grid_cv.fit(X_train , Y_train)




''''
[4] 4. 교차 검증 결과 출력
grid_cv.cv_results_는 교차 검증의 결과를 담고 있는 딕셔너리입니다. 이를 데이터프레임으로 변환하여, 각 하이퍼파라미터 조합에 대한 결과를 확인합니다.
param_max_depth, param_min_samples_split는 각 하이퍼파라미터의 값이고, mean_test_score는 테스트 데이터에서의 평균 정확도, mean_train_score는 훈련 데이터에서의 평균 정확도입니다.
이를 통해 각 하이퍼파라미터 조합에 대한 성능을 비교할 수 있습니다.

[5]5. 최고 정확도와 최적 하이퍼파라미터 출력
grid_cv.best_score_: 교차 검증에서 최고의 성능(정확도)을 기록한 값을 출력합니다.
grid_cv.best_params_: 최적의 성능을 낸 하이퍼파라미터 조합을 출력합니다. 이 하이퍼파라미터 조합이 가장 높은 평균 정확도를 기록한 모델을 의미합니다.

'''
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
print( cv_results_df[['param_max_depth','param_min_samples_split', 'mean_test_score', 'mean_train_score']] )
print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))




'''
[6]6. 최적 모델로 예측 수행
grid_cv.best_estimator_는 가장 성능이 좋았던 모델(최적의 하이퍼파라미터를 가진 결정 트리 모델)을 반환합니다.
best_dt_HAR는 최적의 결정 트리 모델입니다.
이 모델을 사용하여 X_test에 대한 예측을 수행하고, 예측 결과를 best_Y_predict에 저장합니다.
'''
best_dt_HAR = grid_cv.best_estimator_
best_Y_predict = best_dt_HAR.predict(X_test)




'''
[7] 7. 최적 모델의 정확도 계산 및 출력
accuracy_score()를 사용하여 best_Y_predict와 실제 값인 Y_test 간의 정확도를 계산합니다.
최적의 하이퍼파라미터를 가진 결정 트리 모델로 테스트 데이터에 대한 예측 정확도를 계산하고, 이를 소수점 4자리까지 출력합니다.
'''
best_accuracy = accuracy_score(Y_test, best_Y_predict)
print('best 결정 트리 예측 정확도 : {0:.4f}'.format(best_accuracy))





tree.plot_tree(best_dt_HAR, feature_names=feature_name, class_names=label_name, filled=True)
plt.show()





#### **  중요 피처 확인하기
import seaborn as sns
import matplotlib.pyplot as plt

feature_importance_values = best_dt_HAR.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)

feature_top10 = feature_importance_values_s.sort_values(ascending=False)[:10]

plt.figure(figsize = (10, 5))
plt.title('Feature Top 10')
sns.barplot(x=feature_top10, y=feature_top10.index)
plt.show()


from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성.
export_graphviz(best_dt_HAR, out_file="tree.dot", class_names=label_name , feature_names = feature_name,
                impurity=True, filled=True)

import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)



