
### Files and Folders Structure

root
- templates
- - This folder contains HTML templates for the website.
- images 
- - This folder will contain images for the website.
- static
- - This folder contains static files (e.g., CSS, JavaScript) for the website.
- model_training
- - smoke_train.ipynb: Jupyter Notebook file for training smoke detection model.
- - fire_train.ipynb: Jupyter Notebook file for training fire detection model.
- det_models
- - yolov8n.pt: Pre-trained YOLOv8 model for object detection.
- - (other files...): Other model files for detection tasks.
- models
- - fire.py: Python script containing code for fire detection model.
- - smoke.py: Python script containing code for smoke detection model.
- - parking.py: Python script containing code for parking detection model.
- - people.py: Python script containing code for people detection model.
- requirements.txt
- - File listing all Python dependencies required for the project