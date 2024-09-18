import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# 데이터 준비 (데이터를 늘림)
data = {
    'size': [1, 2, 3, 3, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2],
    'color': [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 2, 2, 3, 3],  # 1: 빨간색, 2: 주황색, 3: 노란색
    'labels': [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 2, 2]  # 0: 사과, 1: 오렌지, 2: 바나나
}

df = pd.DataFrame(data)

# 특징과 레이블 분리
x = df[['size', 'color']]
y = df['labels']

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 파라미터 그리드 정의
'''
그리드 서치를 통해 최적의 하이퍼파라미터를 찾기 위한 파라미터 그리드를 정의합니다.
criterion: gini와 entropy 두 가지 기준 중에서 선택.
max_depth: 트리의 최대 깊이. 2, 4, 6 또는 제한 없음(None).
min_samples_split: 분할을 위한 최소 샘플 수. 2, 3, 4.
min_samples_leaf: 리프 노드에 있어야 하는 최소 샘플 수. 1, 2, 3.
'''
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, None],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

# 결정 트리 모델 생성
dt = DecisionTreeClassifier()

# 그리드 서치로 최적의 파라미터 찾기
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
'''
GridSearchCV를 사용하여 5겹 교차 검증(cv=5)으로 그리드 서치를 수행합니다.
fit()을 통해 학습 데이터를 사용하여 최적의 모델을 학습시킵니다.
scoring='accuracy'로 분류 정확도를 기준으로 최적 파라미터를 찾습니다. #????
    scoring='accuracy'는 그리드 서치가 여러 파라미터 조합으로 모델을 훈련한 후, 그 중에서 분류 정확도가 가장 높은 모델을 선택하게 한다는 의미입니다.
    즉, 모든 파라미터 조합 중에서 분류 정확도가 가장 높은 조합을 찾기 위해 정확도를 평가 기준으로 사용한 것입니다.
'''
grid_search.fit(X_train, y_train)

# 최적의 모델로 예측 및 평가
best_model = grid_search.best_estimator_ # ?????
'''
best_estimator_:
베스트 에스티메이터 (영어 발음: /bɛst ɪˈstɪməteɪtər/).
의미:
grid_search.best_estimator_는 최적의 파라미터로 학습된 모델을 의미합니다. 이 모델은 grid_search.fit()이 끝난 후 가장 높은 정확도를 기록한 모델을 가리킵니다. 이 모델을 활용해 새로운 데이터에 대해 예측을 수행할 수 있습니다.
예를 들어, best_model = grid_search.best_estimator_는 최적화된 파라미터로 학습된 모델을 best_model 변수에 저장하는 역할을 합니다.

'''
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"최적의 파라미터: {grid_search.best_params_}")
print(f"최적 모델의 정확도: {accuracy:.2f}")
print(f"트리의 깊이: {best_model.get_depth()}")
print(f"노드의 개수: {best_model.get_n_leaves()}")

# 최적 모델의 결정 트리 구조를 출력 (선택 사항)
tree.plot_tree(best_model, feature_names=['size', 'color'], class_names=['apple', 'orange', 'banana'], filled=True)
plt.show()