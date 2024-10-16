# day33 --> 3.개고양이분류.py

import os
import zipfile
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 압축 파일의 위치 (로컬 파일 경로)
source_filename = "cat-and-dog.zip"  # 로컬 경로
extract_folder = "c:/dataset"  # 압축을 풀 경로

# 압축 해제
with zipfile.ZipFile(source_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# 훈련 셋, 검증 셋 저장위치 지정
train_dir = os.path.join(extract_folder, "archive/training_set/training_set")
valid_dir = os.path.join(extract_folder, "archive/test_set/test_set")

# 이미지 데이터 제너레이터 정의 (Augmentation 미적용)
image_gen = ImageDataGenerator(rescale=(1/255.))

# flow_from_directory 함수로 폴더에서 이미지 가져와서 제너레이터 객체로 정리
train_gen = image_gen.flow_from_directory(train_dir,
                                          batch_size=32,
                                          target_size=(224, 224),
                                          classes=['cats','dogs'],
                                          class_mode = 'binary',
                                          seed=2020)

valid_gen = image_gen.flow_from_directory(valid_dir,
                                          batch_size=32,
                                          target_size=(224, 224),
                                          classes=['cats','dogs'],
                                          class_mode = 'binary',
                                          seed=2020)
# 샘플 이미지 출력
class_labels = ['cats', 'dogs']
batch = next(train_gen)
images, labels = batch[0], batch[1]

plt.figure(figsize=(16, 8))
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(images[i])
    plt.title(class_labels[int(labels[i])])  # int로 변환
    plt.axis("off")
plt.tight_layout()
plt.show()



