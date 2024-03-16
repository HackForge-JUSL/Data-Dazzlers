from ultralytics import YOLO

class fire_detection():
    def __init__(self,model_path,conf=0.85):
        self.model = YOLO(model_path,verbose=False)
        self.confidence = conf

    def process(self,img,flag=True):
        if not flag:
            return (False,[])
        bb_boxes=[]
        result=self.model(img,verbose=False)

        for box in result[0].boxes:
            if(float(box.conf[0])>self.confidence):
                bb=list(map(int,box.xyxy[0]))
                bb_boxes.append(bb)

        if(len(bb_boxes)):
            found=True
        else:
            found=False
        return (found,bb_boxes)