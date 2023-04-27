# Install Flask on your system by writing
#!pip install Flask
#Import all the required libraries
#Importing Flask
#render_template--> To render any html file, template

from flask import Flask, Response,jsonify,request,stream_with_context

# Required to run the YOLOv8 model
import cv2
from ultralytics import YOLO
import math
import json
import time
import os
# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
# from YOLO_Video import video_detection
app = Flask(__name__)
#Generate_frames function takes path of input video file and  gives us the output with bounding boxes
# around detected objects

#Now we will display the output video with detection
# def generate_frames(path_x = ''):
#     yolo_output = video_detection(path_x)
#     for detection_ in yolo_output:
#         ref,buffer=cv2.imencode('.jpg',detection_)
#         frame=buffer.tobytes()
#         yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


stop_capture = dict()

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    global stop_capture
    cap=cv2.VideoCapture(video_capture)
    
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    current_path = os.getcwd()
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
    weight_path =  current_path + "/weights/best.pt"
    model=YOLO(weight_path)
    classNames = ["fire"]
    count = 0
    while True:
        if stop_capture[str(path_x)]:
            break
        if (count % 3 == 0):
            success, img = cap.read()
            if not success: break
            model.classes = classNames
            predict_results = {}
            predict_results['cam_id'] = path_x
            predict_results['fire'] = False
            predict_results['boxes'] = []
            predict_results['probs'] = []

            results=model(img,stream=True)

            for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    conf=math.ceil((box.conf[0]*100))/100
                    predict_results['boxes'].append((x1,y1,x2,y2))
                    predict_results['probs'].append(conf)
            if(len(predict_results['boxes']) > 0):
                predict_results['fire'] = True
            # print(predict_results)
                yield json.dumps(predict_results) + '\n'
            time.sleep(3)
        else:
            count+=1
        
    cap.release()
cv2.destroyAllWindows()


@app.route('/video',methods=["GET"])
def video():
    global stop_capture
    video_path = os.getcwd() + '/video1.mp4'
    stop_capture[video_path] = False
    generator = video_detection(path_x = video_path)
    return Response(generator, content_type='application/json')

@app.route('/ai/<int:cam_id>',methods=["GET"])
def webcam(cam_id):
    global stop_capture
    # if stop_capture.get(str(cam_id)) is None:
    stop_capture[str(cam_id)] = False
    generator = video_detection(path_x = cam_id)
    return Response(generator, content_type='application/json')

@app.route('/stop/<int:cam_id>',methods=["GET"])
def stop(cam_id):
    global stop_capture
    stop_capture[str(cam_id)] = True
    return('Stopped')

    

# @app.route('/',methods=["GET"])
# def vrcam():
#     generator = video_detection(path_x=1)
#     return Response(stream_with_context(generator), content_type='application/json')

if __name__ == "__main__":
    app.run(debug=False)