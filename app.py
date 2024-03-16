import cv2
from flask import Flask,render_template, request,redirect,Response

from multiprocessing import Process


from models.fire import fire_detection
from models.people import people_detection
from models.smoking import smoking_detection
from models.vehicle import vehicle_detection

## loading the models
region_default=[[0,0],[1000,0],[0,1000],[1000,1000]]
model_people=people_detection(model_path="det_models\yolov8n.pt",region=region_default,conf=0.35)
model_vehicle=vehicle_detection(model_path="det_models\yolov8n.pt",region=region_default,conf=0.35)
model_fire=fire_detection(model_path="det_models\\fire.pt",conf=0.45)
model_smoke=smoking_detection(model_path="det_models\smoking.pt",conf=0.45)
print("models_loaded")

def process_frames(camid,
        flag_people=False,
        flag_vehicle=False,
        flag_fire=False,
        flag_smoke=False,
        region=region_default)
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
    if len(camid)==1:
        camid=int(camid)
    cap=cv2.VideoCapture(camid)
    ret=True
    while()
    

