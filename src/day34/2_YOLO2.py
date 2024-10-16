!git clone https://github.com/AlexeyAB/darknet

# GPU 활성화
%cd darknet
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

# Darknet 생성
!make

!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights /content/car.mp4 -out_filename /content/output.avi
