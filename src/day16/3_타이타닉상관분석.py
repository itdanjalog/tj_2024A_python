# day16 > 3_타이타닉상관분석.py

# 1. seaborn 라이브러이 에 내장된 '타이타닉' 데이터를 가져오기
import seaborn as sns
titanic = sns.load_dataset('titanic')
print( titanic )
# 2. 호출된 '타이타닉' 데이터 를 csv 파일로 저장
titanic.to_csv('타이타닉.csv',index=True)
# 3. 결측값(누락된값/공백)
print( titanic.isnull().sum() ) # 결측값 확인
# 4. 결측값을 치환 , # fillna() null(결측)값 특정 값으로 채워주는 함수
    # (1) age 열의 결측값을 중앙값( 크기순으로 정렬된 상태에서 중간에 위치한 값 뜻 )으로 치환
    # median() : 중앙값 반환해주는 함수
titanic['age'] = titanic['age'].fillna( titanic['age'].median() )
print( titanic.isnull().sum() ) # 확인 age에 결측값이 없어졌다.
    # (2) embarked 열의 결측값을 최빈값( 집합의 빈도가 많은 값 ) 으로 치환
print( titanic['embarked'].value_counts() )
titanic['embarked'] = titanic['embarked'].fillna('S')
print( titanic.isnull().sum() ) # 확인 embarked에 결측값이 없어졌다.
    # (3) embark_town 열의 결측값을 최빈값 으로 치환
print( titanic['embark_town'].value_counts() )
titanic['embark_town'] = titanic['embark_town'].fillna('Southampthon')
print( titanic.isnull().sum() ) # 확인 embark_town 결측값이 없어졌다.
    # (4) deck 열의 결측값을 최빈값 으로 치환
print( titanic['deck'].value_counts() )
titanic['deck'] = titanic['deck'].fillna('C')
print( titanic.isnull().sum() ) # 확인 deck 결측값이 없어졌다.

print( titanic.info() )     # 5. 데이터의 기본 정보
print( titanic.survived.value_counts() ) # survived(생존자여부) 속성의 레코드 개수
'''
survived    생존자여부
0    549     0 : 사망자
1    342     1 : 생존자
'''







