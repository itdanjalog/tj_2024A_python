# day34 -> 1_객체탐지.py
import tensorflow as tf
# [1] 샘플 이미지
img_path = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Gangnam_Seoul_January_2009.jpg/1280px-Gangnam_Seoul_January_2009.jpg' # 이미지 경로
# 1. 지정한 경로의 이미지를 파일 객체 로 가져오기
img = tf.keras.utils.get_file( fname='gangnam' , origin = img_path ); print( img )
# 2. 파일 객체를 string 으로 변환
img = tf.io.read_file( img ) ; print( img )
# 3. 문자(string)를 숫자(unit8)텐서 변환
img = tf.image.decode_jpeg( img , channels = 3 ); print( img )
# 4. 0~1 범위로 정규화
img = tf.image.convert_image_dtype( img , tf.float32 ); print( img )
# 5. 시각화
import matplotlib.pyplot as plt
plt.imshow( img )
plt.show()
# 6. 차원 추가
print( img.shape )  # (700, 1280, 3)
img_input = tf.expand_dims( img , 0 ) # 0번 인덱스(가장 앞에) 차원 추가
print( img_input.shape ) # (1, 700, 1280, 3)
# 외부로부터 사전 학습된 모델 가져오기 # 수업에서는 그래픽카드 사양의 한계로 cpu 사용함.
import tensorflow_hub as tfhub # tensorflow-hub 패키지 설치 # python3.8( tf GPU) ---> python3.12( tf CPU)
# 1. 지정한 URL 이용한 모델 로드 하기
model = tfhub.load('https://www.kaggle.com/models/google/faster-rcnn-inception-resnet-v2/tensorFlow1/faster-rcnn-openimages-v4-inception-resnet-v2/1?tfhub-redirect=true');
print( model.signatures.keys() )  # 모델 시그너처(용도) 확인
obj_detector = model.signatures['default'] # 주어진 key(사용가능한 변수/객체는 default ) 만 존재한다.
print( obj_detector )
# 2. 로드한 모델로 예측하기
result = obj_detector( img_input )
print( result.keys() ) # 경계박스 좌표 , 예측한/검출된 클래스(정답/종속) 아이디 , 예측/검출된 확률/스코어
print( len( result['detection_scores'] )  )



