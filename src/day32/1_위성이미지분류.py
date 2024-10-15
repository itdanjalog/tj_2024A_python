# day32 --> 1.위성이미지분류.py
# 딥러닝 프로세스
# (1) 데이터 수집
# (2) 데이터 전처리 / 데이터 분할
# (3) 모델 설계(구축)
# (4) 모델 컴파일
# (5) 모델 학습 ----> 모델 튜닝(최적의 하이퍼 파라미터 찾기) -> (3)
# (6) 모델 평가 및 예측

#
import tensorflow as tf
import tensorflow_datasets as tfds # tensorflow-datasets 설치
(train_ds , valid_ds) , info  = tfds.load( 'eurosat/rgb' , # 다운로드 할 데이터셋 이름
           split=['train[ :80%]' , 'train[80:]'] , # 80%의 데이터를 훈련용 , 20%를 검증용으로 분할하기.
           shuffle_files=True , # 파일을 무작위로 섞어 데이터를 로드한다.
           as_supervised=True , # 이미지와 레이블로 구성된 튜플로 가져오기.
           with_info = True ,  # 데이터셋의 메타정보(데이터셋설명) 가져오기.
           data_dir = 'dataset/' # 현재 py 파일이 위치한 폴더내 하위 폴더로 'dataset' 폴더안에 '데이터셋' 를 다운로드 하겠다.
           )

print( train_ds )
print( valid_ds )
print( info )


