from ultralytics import YOLO
import cv2

class people_model():
    """
    this class will be used to detect if people are in 
    no people zone.

    Args:
    model_path: path to model
    region: list containg regoin coordinates in 
            [(x1,y1),(x2,y2)] format
    conf: minimum confidence to consider 
    
    """

    def __init__(self,model_path,region,conf=0.85):
        self.model=YOLO(model_path,verbose=False)
        self.region=region
        self.conf=conf