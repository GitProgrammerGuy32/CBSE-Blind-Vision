from flask import Flask, render_template, Response
import cv2
import urllib.request
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Object detection setup (same as before)
classNames = []
classFile = "Weights/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "Weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Weights/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Face detection setup
face_model_path = 'Weights/facev3.tflite'
face_label_path = 'Weights/face.txt'
face_min_confidence = 0.7

face_interpreter = Interpreter(model_path=face_model_path)
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()
face_height, face_width = face_input_details[0]['shape'][1], face_input_details[0]['shape'][2]

face_float_input = (face_input_details[0]['dtype'] == np.float32)
face_input_mean, face_input_std = 127.5, 127.5

with open(face_label_path, 'r') as f:
    face_labels = [line.strip() for line in f.readlines()]

# Common camera URL for both object detection and face detection
camera_url = 'http://192.168.1.8/cam-hi.jpg'  # Update with your camera URL

def generate(url, detection_type):
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        if detection_type == 'object':
            result, _ = getObjects(frame, 0.60, 0.2)
        elif detection_type == 'face':
            result = detectFaces(frame)

        _, jpeg = cv2.imencode('.jpg', result)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

def detectFaces(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (face_width, face_height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if face_float_input:
        input_data = (np.float32(input_data) - face_input_mean) / face_input_std

    face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
    face_interpreter.invoke()

    boxes = face_interpreter.get_tensor(face_output_details[1]['index'])[0]
    classes = face_interpreter.get_tensor(face_output_details[3]['index'])[0]
    scores = face_interpreter.get_tensor(face_output_details[0]['index'])[0]

    for i in range(len(scores)):
        if face_min_confidence < scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = face_labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object_detection_video')
def object_detection_video():
    return Response(generate(camera_url, 'object'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_detection_video')
def face_detection_video():
    return Response(generate(camera_url, 'face'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/testface')
def sample():
    return "<h1>BITCH</h1>"

if __name__ == "__main__":
    app.run(debug=True)
