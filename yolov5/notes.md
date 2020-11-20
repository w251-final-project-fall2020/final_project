docker run --privileged --runtime nvidia --rm  -v /data:/data -e DISPLAY -v /tmp:/tmp -v /home/surya/.aws:/home/surya/.aws -v /home/surya/final_project/jetson:/usr/src/app -ti final_project python3 detect.py --source 0 --weights yolov5x.pt --conf 0.4


docker run --privileged --shm-size 50G -e DISPLAY=$DISPLAY --rm  -v /tmp:/tmp -v /home/surya/MIDS/251/final_project/yolo/runs:/usr/src/app/runs -ti fp python3 train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt 

docker run --privileged  -e DISPLAY=$DISPLAY --runtime nvidia --rm  -v /tmp:/tmp -v /home/surya/final_project/coco-mids:/usr/src/app/coco-mids  -ti yolov5 python3 train.py --batch 16 --epochs 5 --data /usr/src/app/final-project/yolov5/data/
coco-mids.yaml --weights /usr/src/app/final-project/yolov5/weights/yolov5s.pt 



docker run --privileged --shm-size 50G -e DISPLAY=$DISPLAY --rm  -v /home/surya/MIDS/251/final_project/yolo/runs:/usr/src/app/runs -v /tmp:/tmp -ti fp python3 train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt 


docker run --privileged --shm-size 50G -e DISPLAY=$DISPLAY --rm  -v /home/surya/MIDS/251/final_project/yolo/runs:/usr/src/app/runs -v /home/surya/MIDS/251/final_project/coco-mids:/usr/src/coco-mids -v /tmp:/tmp -ti fp python3 train.py --img 640 --batch 16 --epochs 5 --data coco-mids.yaml --weights yolov5s.pt 
