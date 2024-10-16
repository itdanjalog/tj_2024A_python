from yolov5 import YOLOv5

# YOLOv5 모델 로드
model = YOLOv5('yolov5s.pt')  # 사전 훈련된 가중치 파일 경로

# 이미지 예측
results = model.predict('carimg.jpg')
results.show()  # 결과 이미지 표시
