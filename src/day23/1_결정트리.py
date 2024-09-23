# day23 -> 1_결정트리.py
# 결정트리 : 결정트리(다중분류) vs 로지스틱 회귀(이진분류)
# 모델 생성 하고 예측
# [1] 데이터 수집 # 데이터셋 찾는 과정
# 스마트폰으로 수집한 사람의 움직임 데이터
# 1. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
# 2. [다운로드] UCI HAR Dataset 폴더
# 3. 피처(독립변수) 이름 파일 읽어오기 # txt 파일 읽어오기
import pandas as pd
feature_name_df = pd.read_csv('features.txt' , sep='\s+' , header=None , names=['index' , 'feature_name'] , engine='python' )
    # 1.  sep='\s+' : 공백으로 구분된 형식 파일  # 2. header=None : 제목이 없는 파일  # 3. names= [ 열이름 ] # 4. engine='python' 생략가능
print( feature_name_df.head() )
print( feature_name_df.shape ) # (561, 2)
# 4. 인덱스 제거 , 독립변수 이름만 리스트에 저장 # 두번째 열의 모든 행을 리스트로 반환
feature_name = feature_name_df.iloc[ : , 1 ].values.tolist()
    # 데이터프레임객체.iloc[ 행 슬라이싱 ]
    # 데이터프레임객체.iloc[ 행 슬라이싱 , 열번호 ]
    # feature_name_df.iloc[ : ] : 모든 행   # feature_name_df.iloc[ : , 1 ] : 모든 행의 두번째 열 ( 첫번째 열 제외 )
    # .values 열의 모든 값들을 추출 # .tolist() 리스트로 반환 함수
print( feature_name )