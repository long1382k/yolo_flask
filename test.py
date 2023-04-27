from ultralytics import YOLO
import math
import cv2
import numpy as np

img = r'D:\Workspace\development\fire_detection_flask\code2\yolov5-flask\tests\000004.jpg'
model_path = "D:/Workspace/development/fire_detection_flask/code2/yolov5-flask/weights/best.pt"
batch = img
predict_results = dict()
predict_results['imgs'] = []
model = YOLO(model_path)
results = model(img,save=True)
img = cv2.imread(img)
for r in results:
    boxes=r.boxes
    for box in boxes:
        x1,y1,x2,y2=box.xyxy[0]
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        conf=math.ceil((box.conf[0]*100))/100
        predict_results['imgs'].append(img[y1:y2,x1:x2])
print(predict_results)

