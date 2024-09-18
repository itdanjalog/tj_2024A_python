import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# 데이터 준비
data = {
    'size': [1, 2, 3, 3, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2],
    'color': [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 2, 2, 3, 3],  # 1: 빨간색, 2: 주황색, 3: 노란색
    'labels': [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 2, 2]  # 0: 사과, 1: 오렌지, 2: 바나나
}
df = pd.DataFrame( data )

x = df[ ['size','color'] ]
y = df['labels']

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(x , y , test_size=0.2, random_state=42 )


# 결정 트리 모델 생성 ( 디시전 트리 클래시파이어 )
model = DecisionTreeClassifier(  )
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"트리의 깊이: {model.get_depth()}")
print(f"노드의 개수: {model.get_n_leaves()}")
print(f"모델의 정확도: {accuracy:.2f}")

# 모델의 결정 트리 구조를 출력 (선택 사항)
tree.plot_tree(model, feature_names=['size', 'color'], class_names=['apple', 'orange' ,'banana'], filled=True)
plt.show()
