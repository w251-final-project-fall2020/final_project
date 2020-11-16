docker run --privileged --runtime nvidia --rm  -v /data:/data -e DISPLAY -v /tmp:/tmp -v /home/surya/final_project/jetson:/usr/src/app -ti final_project python3 detect.py --source 0 --weights yolov5x.pt --conf 0.4

docker run --privileged --runtime nvidia --rm  -v /data:/data -e DISPLAY -v /tmp:/tmp -ti final_project python3 detect.py --source 0 --weights yolov5x.pt --conf 0.4