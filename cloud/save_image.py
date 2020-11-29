import numpy as np
import sys

import paho.mqtt.client as mqtt
from io import BytesIO
import boto3
from datetime import datetime
from PIL import Image

LOCAL_MQTT_HOST = 'mosquitto'
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = 'food_detector_cloud'

bucket = boto3.resource('s3').Bucket('food-detector')

dynamodb = boto3.client("dynamodb")

def extractData(payload):
  image_str, save_timestamp, index, num_items, label, confidence, weight = payload.decode('utf-8').split(',')
  print(image_str)
  image_arr = eval('np.array(' + image_str + ')')
  image_bytes = Image.fromarray(image_arr, 'RGB').tobytes()
  return image_bytes, save_timestamp, index, num_items, label, confidence, weight
  

def on_connect_local(client, userdata, flags, rc):
  print("connected to local broker with rc: " + str(rc))
  client.subscribe(LOCAL_MQTT_TOPIC)

def on_message(client, userdata, msg):

  try:
    print("message received!")
    
    image, save_timestamp, index, num_items, label, confidence, weight = extractData(msg.payload)

    #save picture to s3
    stream = BytesIO(image)
    stream.seek(0) # rewind pointer back to start

    filename = save_timestamp + '_' + index + '.png'

    bucket.put_object(
      Key=filename,
      Body=stream,
      ContentType='image/png'
    )

    response = dynamodb.put_item(
        TableName='food-detector', 
        Item={
            "label": label,
            "confidence": confidence,
            "total_items": num_items,
            "total_weight": weight,
            "image_filepath": filename,
            "save_timestamp": save_timestamp
        }
    )

  except:
    print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()