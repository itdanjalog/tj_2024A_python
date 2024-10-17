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
# 3. 모델이 예측한 결과를 시각화
boxes = result['detection_boxes'] # 좌표 예측값
labels = result['detection_class_entities'] # 클래스(정답/종속) 아이디
scores = result['detection_scores'] #  예측/검출된 확률/스코어
print( boxes );  print( labels ); print( scores );
# 샘플 이미지 가로 세로 크기
img_height = img.shape[0] # 가로
img_width = img.shape[1] # 세로
obj_to_detect = 10 # 탐지할 최대 객체수
print(min( obj_to_detect , boxes.shape[0] )  )
for i in range( min( obj_to_detect , boxes.shape[0] ) ) :
    if scores[i] >= 0.2 : # 예측 신뢰도 가 0.2( 20%) 이상이면
        ymax = boxes[i][0] * img_height    # 사각 박스의 위치 좌표 : 상단 끝
        xmin = boxes[i][1] * img_width    # 사각 박스의 위치 좌표 : 왼쪽 끝
        ymin = boxes[i][2] * img_height    # 사각 박스의 위치 좌표 : 하단 끝
        xmax = boxes[i][3] * img_width    # 사각 박스의 위치 좌표 : 오른쪽 끝
        plt.imshow( img ) # 차트에 이미지 넣기
        # 예측한 경계 상자를 그리기
        plt.plot( [xmin , xmax , xmax , xmin , xmin ] , [ ymin , ymin , ymax , ymax , ymin] , color = 'yellow' , linewidth = 2 )
        # 네 점의 좌표 찍기 : ( xmin , ymin ) 왼쪽하단꼭지점, (xmax,ymin)오른쪽하단꼭지점 , (xmax,ymax)오른쪽상단꼭지점 ~~ 연결 경계선를 노란색
        # 4개의 좌표를 그리고 마지막으로 첫번째 좌표를 한번더 명시하여 마침을 한다. ( 다섯번째 점이 첫번째 점과 같아야 도형 닫기가 가능하다. )
        class_name = labels[i].numpy().decode('utf-8') # 예측 레이블 utf-8 인코딩
        infer_score = int( scores[i].numpy() * 100 ) # 신뢰도 백분율
        annotation = f'{class_name}:{infer_score}%'
        plt.text( xmin+10 , ymax+20 , annotation , color = 'white' , backgroundcolor='blue' , fontsize = 10 ) # 왼쪽 상단
plt.show() # 차트 열기


















