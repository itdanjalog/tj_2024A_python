# day33 --> 2.위성이미지분류.py
# 1. matplot
# 2. pandas
# 3. flask
# 4. scikit-learn
# 5. tensorflow-datasets ( pip 최신버전 재설치 )

# (1) 데이터 수집
import tensorflow as tf
import tensorflow_datasets as tfds # tensorflow-datasets 설치
(train_ds , valid_ds) , info  = tfds.load( 'eurosat/rgb' , # 다운로드 할 데이터셋 이름
           split=['train[:80%]' , 'train[80%:]'] , # 80%의 데이터를 훈련용 , 20%를 검증용으로 분할하기.
           shuffle_files=True , # 파일을 무작위로 섞어 데이터를 로드한다.
           as_supervised=True , # 이미지와 레이블로 구성된 튜플로 가져오기.
           with_info = True ,  # 데이터셋의 메타정보(데이터셋설명) 가져오기.
           data_dir = 'c:/dataset/' # 현재 py 파일이 위치한 폴더내 하위 폴더로 'dataset' 폴더안에 '데이터셋' 를 다운로드 하겠다.
           )
print( train_ds )
print( valid_ds )
print( info )
# .show_examples() : 샘플의 이미지와 분류 레이블 출력해주는 함수
tfds.show_examples( train_ds , info)
# .as_dataframe()
print( tfds.as_dataframe( valid_ds.take(10) , info ) )
# 전체 레이블 수 확인
NUM_CLASSES = info.features['label'].num_classes
# 특정 번호의 레이블 확인
print( info.features['label'].int2str(6) ) # PermanentCrop : 영구작물
# 0:경작지 1.숲 2.식물 3.고속도로 4.산업지역 5.목초지  6:영구작물 7.주거지역 8.강 9:바다/호수

# (2) 데이터 전처리
BATCH_SIZE = 64 # 배치란? 한번에 처리하는 데이터의 묶음 단위 의미한다.
    # 데이터를 배치로 나눠서 처리하면 메모리 사용을 최적할수 있다.
    # 모델이 전체를 한번에 처리하지 않고 데이터를 묶음(배치) 단위로 나누어 처리한다. - 배치 처리
BUFFER_SIZE = 1000 # 버퍼란? 임시 저장공간
    # 셔플 할때 버퍼에 데이터를 1000를 가져와서 임시로 저장하는 공간이다.
    # 셔플 : 일반적으로 정형화된 데이터들을 순서대로 넣으면 모델의 특정 패턴이 치우치게 될수 있기 때문에 섞어준다.

def preprocess_data( image , label ) :
    image = tf.cast( image , tf.float32 ) / 255.0 # 이미지타입을 float32 변환하고 # 0~255 -> 0~1 정규화
    return  ( image , label ) # 이미지와 레이블을 튜플구조로 반환하기
# JS map 함수
# newArray = [ 3 , 2 , 1 ].map( ( value )->{ return value + 10  } )
# newArray( 13 , 12 , 11 )

# num_parallel_calls = tf.data.AUTOTUNE : 병렬처리(병렬 매핑)
train_data = train_ds.map(preprocess_data , num_parallel_calls = tf.data.AUTOTUNE )
valid_data = valid_ds.map(preprocess_data , num_parallel_calls = tf.data.AUTOTUNE )

# 훈련용데이터를 셔플링(가중치-업데이트o) 하고 캐시(기록) 제외한  오토튠을 적용했다.
# 검증용데이터를 셔플링(가중치-업데이트x) 하지 않고 캐시(기록) 하고  오토튠을 적용했다.
train_data = train_data.shuffle( BUFFER_SIZE ).batch( BATCH_SIZE ).prefetch( tf.data.AUTOTUNE )
valid_data = valid_data.batch( BATCH_SIZE ).cache().prefetch( tf.data.AUTOTUNE )
# cache() : 캐시( 기록 )란 검증데이터셋 메모리를 캐시한다.
# 한번 호출한 검증데이터는 메모리에 기록하여 다음에 호출시 빠르게 접근할수 있도록 하는 함수

# prefetch( tf.data.AUTOTUNE ) : 데이터 전처리 와 훈련을 병렬로 수행하여 학습 속도를 향상 할수 있다.

# 모델링 함수
def build_model():
    model = tf.keras.Sequential([ # Sequential Api 이용한 모델 구축
        # =========================== [ Convolution 층 = 합성곱 = 연산층 = 특징 찾기  ]
        # 1. 합성곱(신경망) 레이어
        tf.keras.layers.BatchNormalization(),
        # 배치 : 모델링에 있어서 병렬처리에 배치(묶음) 단위로 처리하면 더 빠르고 안정적으로 학습할수있다. # 과대적합을 줄이기 가능하다.
        # BatchNormalization() 레이어가 없어도 모델 구현이 가능하지만 모델의 최적화에 필요한 레이어 # 주로 AUTOTUNE 사용시 사용된다.
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 1. 합성곱(신경망) 레이어  # 복잡한 신경망 구현 하기 위해 2번의 합성곱을 실행했다.
        # 1번만 합성곱 연산으로 모델 구축 가능 하지만 더 많은 특징을 찾기 위해 2번의 레이어 만들었다.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # =========================== [ Classifier 층 = 출력층 = 예측분류층 = 특징 학습  ]
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'), # Dense : 완전 연결된 신경망 층 , 이전 node(뉴런) 받아서 패턴 학습 한다.
        # 주로 노드 개수 는 32 64 128 사용한다.
        tf.keras.layers.Dropout(0.3),                   # 드롭아웃[ p.89 ] : 30% 노드를 랜덤으로 제외하고 학습 하는 설정 레이어
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'), # 출력층
    ])
    return model
model = build_model()
# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 모델 훈련
history = model.fit( train_data,  validation_data=valid_data,  epochs = 3 )

# 손실함수 , 정확도 그래프 그리기 # 교재에서 제외된 부분
import matplotlib.pyplot as plt
def plot_loss_acc( history , epoch  ) :
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    fig , axes  = plt.subplots( 1 , 2 )
    axes[0].plot( range(1 , epoch + 1 ) , loss )
    axes[0].plot( range(1, epoch + 1) , val_loss)
    axes[0].set_title('Loss')

    axes[1].plot( range(1 , epoch + 1) , acc )
    axes[1].plot( range(1 , epoch + 1) , val_acc )
    axes[1].set_title('Accuracy')

    plt.show()

plot_loss_acc( history , 3 )


















