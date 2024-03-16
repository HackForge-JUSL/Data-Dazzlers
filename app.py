import cv2


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
    if  (len(camid)==1):
        camid=int(camid)
    
    cap=cv2.VideoCapture(camid)
    ret=True
    while(True):
        ret,frame=cap.read()
        if not ret:
            break

        found_fire,bb_box_fire=model_fire.process(frame,flag=flag_fire)
        if found_fire:
            for box in bb_box_fire:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)

        found_people,bb_box_people=model_people.process(frame,flag=flag_people,region=region)
        if found_people:
            for box in bb_box_people:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)

        found_smoke,bb_box_smoke=model_smoke.process(frame,flag=flag_smoke)
        if found_smoke:
            for box in bb_box_smoke:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)

        found_vehicle,bb_box_vehicle=model_vehicle.process(img=frame,flag=flag_vehicle,region=region)
        if found_vehicle:
            for box in bb_box_vehicle:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)


        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
if __name__=="__main__":
    print("good run")
    

