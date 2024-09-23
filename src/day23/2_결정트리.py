# day23 > 2_결정트리.py
# 주제 : 여러 어종의 특성(Weight,Length,Diagonal,Height,Width)들을 바탕으로 어종명(Species) 예측 하기
# Species : 어종명 , Weight:무게 , Length:길이 , Diagonal:대각선길이 , Height:높이 , Width:너비
# 어종 데이터셋 : https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv
# [1] 데이터 셋
import pandas as pd
data = pd.read_csv( 'https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv' )
print( data.head() ) # 확인
# [2] 7:3 비율로 훈련용 과 테스트용으로 분리 하기
# [3] 결정트리 모델로 훈련용 데이터 피팅 하기
# [4] 훈련된 모델 기반으로 테스트용 데이터 예측 하고 예측 정확도 확인하기
# 출력예시 ]    개선 전 결정트리모델 정확도 : 0.xxx
# [5] 최적의 하이퍼 파라미터 찾기 # params = { 'max_depth' : [ 2 , 6 , 10 , 14 ] , 'min_samples_split:[ 2 , 4 , 6 , 8 ] }
# 출력예시 ]    평균 정확도 : x.xxxxxxx  , 최적 하이퍼파라미터 : { 'max_depth' : xx , 'min_samples_split': x  }
# [6] 최적의 하이퍼 파라미터 기반으로 모델 개선후 테스트용 데이터 예측하고 예측 정확도 학인하기 # 시각화하기
# 출력예시 ]    개선 후 결정트리모델 정확도 : 0.xxx
# 차트 시각화

# [제출] : 콘솔 출력값이 시각화 같이 보이도록 캡처후 카톡방에 제출






