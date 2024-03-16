from ultralytics import YOLO


class no_vehicle_zone():
    """
    this class will be used to detect if vehicle are in 
    no parking zone.

    Args:
    model_path: path to model
    region: list containg regoin coordinates in 
            [(x1,y1),(x2,y2)] format
    conf: minimum confidence to consider detection


    """

    def __init__(self,model_path,region,conf=0.75):

        """
        basic inti function
        """
        self.model=YOLO(model_path,verbose=False)
        self.region=region
        self.conf=conf

    def in_region(self, point, region):
        """
        function to check if the point is in region
        """
        x, y = point
        x1, y1, x2, y2 = region
        return x1 <= x <= x2 and y1 <= y <= y2
