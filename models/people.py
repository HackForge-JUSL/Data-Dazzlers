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
    conf: minimum confidence to consider detection 
    
    """

    def __init__(self,model_path,region,conf=0.85):
        """
        basic inti function
        """
        self.model=YOLO(model_path,verbose=False)
        self.region=region
        self.conf=conf

    def in_region(self, point):
        """
        this function checks is the given point is in region
        """
        x, y = point
        x1, y1, x2, y2 = self.region
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def process(self,img):
        """
        this function processes the cv2 frame and returns the
        bounding boxes
        """

        bb_boxes=[]
        results=self.model(img,verbose=False)

        for box in results[0].boxes:
            if (int(box.cls[0])==0 and float(box.conf[0])>self.conf):
                bb=list(map(int,box.xyxy[0]))
                center=[(bb[0]+bb[2])//2,(bb[1]+bb[3])//2]

                if(self.in_region(center)):
                    bb_boxes.append((True,bb))

        if not len(bb_boxes):
            bb_boxes.append((False,[]))
        
        return bb_boxes
