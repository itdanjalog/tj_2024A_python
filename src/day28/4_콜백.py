import tensorflow as tf

# 케라스의 내장 데이터셋에서 mnist 데이터셋을 로드
mnist = tf.keras.datasets.mnist

# load_data()로 데이터셋을 로드 합니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), # 노드 10개로 생성
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 체크포인트 설정
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='tmp_checkpoint.weights.h5',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                verbose=1)

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=10,
          callbacks=[checkpoint]
          )


# 모델 체크포인트 로드 전
loss, acc = model.evaluate(x_test, y_test)
print(f'체크포인트 로드 전: loss: {loss:.4f}, acc: {acc:.4f}')

# 체크포인트 파일을 모델에 로드
model.load_weights('tmp_checkpoint.weights.h5')
loss, acc = model.evaluate(x_test, y_test)
print(f'체크포인트 로드 후: loss: {loss:.4f}, acc: {acc:.4f}')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), # 노드 10개로 생성
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 생성
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=20,
          callbacks=[earlystopping]
          )

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), # 노드 10개로 생성
])

def scheduler(epoch, lr):
    tf.print(f'learning_rate: {lr:.5f}')
    # 첫 5 에포크 동안 유지
    if epoch < 5:
        return float(lr)
    else:
    # 학습률 감소 적용
        return float(lr * tf.math.exp(-0.1))

# 콜백 객체생성 및 scheduler 함수 적용
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 초기 학습률 확인(0.01)
print(round(model.optimizer.learning_rate.numpy(), 5))

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=10,
          # 학습률 스케줄러 적용
          callbacks=[lr_scheduler]
          )
# 최종 학습률 스케줄러 확인
round(model.optimizer.learning_rate.numpy(), 5)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), # 노드 10개로 생성
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 텐서보드 저장 경로 지정
log_dir = 'tensorboard'

# 텐서보드 콜백 정의
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=10,
            callbacks=[tensorboard],
            )

# # 텐서보드 extension 로드
# %load_ext tensorboard
#
# # 텐서보드 출력 매직 커멘드
# %tensorboard --logdir {log_dir}