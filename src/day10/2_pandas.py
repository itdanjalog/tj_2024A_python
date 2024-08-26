import pandas as pd
print( pd.__version__)

# 1.
data1 = [ 10 ,20,30,40,50]
print( data1 )

data2 = ['1반','2반','3반','4반','5반']
print( data2 )

# 2.
sr1 = pd.Series( data1 )
print( sr1 )

sr2 = pd.Series( data2 )
print( sr2 )

# 4.
sr3 = pd.Series( [101,102,103,104,105] )
print( sr3 )

sr4 = pd.Series( ['월','화','수','목','금'] )
print( sr4 )

# 5.
sr5 = pd.Series( data1 , index=[1000,1001,1002,1003,1004])
print( sr5 )

sr6 = pd.Series( data1 , index= data2 )
print( sr6 )

sr7 = pd.Series( data2 , index= data1 )
print( sr7 )

sr8 = pd.Series( data2 , index= sr4 )
print( sr8 )

# 6.
print( sr8.iloc[2] ) # ????
print( sr8['수'] )
print( sr8.iloc[-1] )
#
print( sr8[0:4] )
#
print( sr8.index )
#
print( sr8.values )
print( sr1 + sr3 )

print( sr4 + sr2 )

#
data_dic = {
    'year' : [2018,2019,2020],
    'sales' : [350,480,1099]
}
print( data_dic )
df1 = pd.DataFrame( data_dic )
print( df1 )

#
df2 = pd.DataFrame( [[89.2,92.5,90.8],[92.8,89.9,95.2]] , index=['중간고사','기말고사'] , columns=data2[0:3] )
print( df2 )
#
data_df = [['20201101','Hong', '90' , '95'], ['20201102','kim','93','94'],['20201103','Lee','87','97']]
df3 = pd.DataFrame( data_df )
print( df3 )

#
df3.columns = [ '학번' , '이름' , '중간고사' , '기말고사' ]

print( df3 )

#
print( df3.head(2) )

print( df3.tail(2) )

print( df3['이름'] )

#
df3.to_csv( 'score.csv' , header='False'  ) # index ?????

#
df4 = pd.read_csv( 'score.csv' , encoding='utf-8' , index_col=0 , engine='python')
print( df4 )

#

# 데이터 생성
data = {
    '이름': ['김철수', '이영희', '박준호'],
    '나이': [28, 34, 29],
    '직업': ['엔지니어', '디자이너', '데이터 분석가']
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 데이터프레임 출력
print("데이터프레임:")
print(df)

# 열의 평균값 계산
average_age = df['나이'].mean()
print(f"\n평균 나이: {average_age:.2f}")

# 특정 조건에 맞는 데이터 필터링
filtered_df = df[df['나이'] > 30]
print("\n나이가 30세 이상인 사람들:")
print(filtered_df)

# 새로운 열 추가
df['연봉'] = [50000, 60000, 55000]
print("\n연봉이 추가된 데이터프레임:")
print(df)

# 데이터프레임을 CSV 파일로 저장
df.to_csv('example.csv', index=False)

# CSV 파일에서 데이터프레임 불러오기
loaded_df = pd.read_csv('example.csv')
print("\nCSV 파일에서 불러온 데이터프레임:")
print(loaded_df)






















