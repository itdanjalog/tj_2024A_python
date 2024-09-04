import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic")
titanic.to_csv('titanic.csv', index = False)
#### - 결측값 있는지 확인하기
print( titanic.isnull().sum() )
#### - age 열의 결측값을 중앙값(크기 순으로 정렬했을 때, 중간에 위치하는 값)으로 치환하기
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
#### - embarked 열의 결측값을 최빈값(데이터 집합에서 가장 자주 나타나는 값) 으로 치환하기
titanic['embarked'].value_counts()
titanic['embarked'] = titanic['embarked'].fillna('S')
titanic['embark_town'].value_counts()
titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')
titanic['deck'].value_counts()
titanic['deck'] = titanic['deck'].fillna('C')

print( titanic.isnull().sum() )

###########
print( titanic.info() )
print( titanic.survived.value_counts())

#############
import matplotlib.pyplot as plt
f,ax = plt.subplots(1, 2, figsize = (10, 5))

titanic['survived'][titanic['sex'] == 'male'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
# explode=[0,0.1]는 두 조각 중 두 번째 조각(생존하지 않은 경우)을 약간 떨어뜨려 강조합니다.
titanic['survived'][titanic['sex'] == 'female'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow = True)
ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (Female)')

plt.show()

##################
sns.countplot(x='pclass', hue = 'survived', data = titanic)
plt.title('Pclass vs Survived')
plt.show()

#################
titanic2 = titanic.select_dtypes(include=[int, float,bool])
print( titanic2.shape )
titanic_corr = titanic2.corr(method = 'pearson')
print( titanic_corr )
titanic_corr.to_csv('titanic_corr.csv', index = False)
#################
titanic['survived'].corr(titanic['adult_male'])
titanic['survived'].corr(titanic['fare'])
################
sns.pairplot(titanic, hue = 'survived')
plt.show()

##################
sns.catplot(x = 'pclass', y = 'survived', hue = 'sex', data = titanic, kind = 'point')
plt.show()

##################
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7
titanic['age2'] = titanic['age'].apply(category_age)
titanic['sex'] = titanic['sex'].map({'male':1, 'female':0})
titanic['family'] = titanic['sibsp'] + titanic['parch'] + 1
titanic.to_csv('titanic3.csv', index = False)
heatmap_data = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]

colormap = plt.cm.RdBu

sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True,
annot_kws = {"size": 10})

plt.show()
