import cv2
import numpy as np
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)


from models.fire import fire_detection
from models.people import people_detection
from models.smoking import smoking_detection
from models.vehicle import vehicle_detection

## loading the models
region_default=[[0,0],[1000,0],[0,1000],[1000,1000]]
model_people=people_detection(model_path="det_models\yolov8n.pt",conf=0.35)
model_vehicle=vehicle_detection(model_path="det_models\yolov8n.pt",conf=0.35)
model_fire=fire_detection(model_path="det_models\\fire.pt",conf=0.45)
model_smoke=smoking_detection(model_path="det_models\smoking.pt",conf=0.45)
print("models_loaded")

def process_frames(camid,region,flag_people=False,flag_vehicle=False,flag_fire=False,flag_smoke=False):
    """
    function to process frames

    Args:
    camid: camid given by the user 
    flag_people: Flag for peolpe_detection 
    flag_vehicle: Flag for vehicle detection
    flag_fiere: Flag for fire detection 
    flag_smoke: Flag for smoke detection

    returns: image object

    """
    ##naya121004if  (len(camid)==1):
    ##    camid=int(camid)
    
    cap=cv2.VideoCapture(camid)
    ret=True
    while(True):
        ret,frame=cap.read()
        if not ret:
            break

        frame=cv2.resize(frame,(150,200))
        frame=cv2.polylines(frame,[np.array(region).reshape(-1,1,2)],True,(0,0,255),1)

        """found_fire,bb_box_fire=model_fire.process(frame,flag=flag_fire)
        if found_fire:
            for box in bb_box_fire:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)"""

        #print(flag_people)
        found_people,bb_box_people=model_people.process(frame,flag=flag_people,region=region)
        ##print(found_people,bb_box_people)
        if found_people:
            for box in bb_box_people:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)

        """found_smoke,bb_box_smoke=model_smoke.process(frame,flag=flag_smoke)
        if found_smoke:
            for box in bb_box_smoke:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)

        found_vehicle,bb_box_vehicle=model_vehicle.process(img=frame,flag=flag_vehicle,region=region)
        if found_vehicle:
            for box in bb_box_vehicle:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)"""


        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

@app.route('/')
def index():
    return render_template('dash.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(camid=0,region=region_default,flag_people=True), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

