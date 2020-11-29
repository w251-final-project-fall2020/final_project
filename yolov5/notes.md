docker network create --driver bridge final_project




inference (jetson):

docker network create --driver bridge final_project

docker run --name mosquitto --network final_project -p 1883:1883 -ti alpine sh

    apk update && apk add mosquitto
    /usr/sbin/mosquitto


docker run --privileged --runtime nvidia --network final_project --rm  -v /data:/data -e DISPLAY -v /tmp:/tmp -v /home/surya/final_project/yolov5:/usr/src/app -ti final_project python3 detect.py --source 0 --weights runs/train/exp5_x.16.20/weights/best.pt --conf 0.1


cloud stuff:

aws ec2 describe-vpcs | grep VpcId
            "VpcId": "vpc-9600faeb",

aws ec2 create-security-group --group-name final_project --description "E2E Pipeline - Cloud" --vpc-id vpc-9600faeb
{
    "GroupId": "sg-03d813fab1748d786"
}

aws ec2 authorize-security-group-ingress --group-id sg-03d813fab1748d786 --protocol tcp --port 22 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress --group-id sg-03d813fab1748d786 --protocol tcp --port 1883 --cidr 0.0.0.0/0

aws ec2 describe-images  --filters  Name=name,Values='ubuntu/images/hvm-ssd/ubuntu-bionic-18.04*' Name=architecture,Values=x86_64   | head -100

    ami-00a208c7cdba991ea

aws ec2 run-instances --image-id ami-00a208c7cdba991ea --instance-type t2.micro --security-group-ids sg-03d813fab1748d786 --associate-public-ip-address --key-name desktop-east-1q

aws ec2 describe-instances | grep PublicDnsName
        
        "PublicDnsName": "ec2-3-89-113-213.compute-1.amazonaws.com",

ssh -i desktop-east-1.pem ubuntu@ec2-3-89-113-213.compute-1.amazonaws.com

docker network create --driver bridge final_project_cloud

docker run --name mosquitto --network final_project_cloud -p 1883:1883 -ti alpine sh

docker build -t image_saver -f Dockerfile .

docker run --privileged --rm -v /home/ubuntu/.aws:/root/.aws --name image_saver --network final_project_cloud -ti image_saver python3 save_image.py