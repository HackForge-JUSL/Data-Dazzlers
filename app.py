import cv2
from flask import Flask,render_template, request,redirect,Response



app = Flask(__name__)

cameras={}


@app.route('/')
def dashboard():
    return render_template('dashboard.html',cameras=cameras)


def add_camera():
    cam_id = request.form['camId']
    indoor = 'indoor' in request.form
    people_detection = 'peopleDetection' in request.form
    fire_detection = 'fireDetection' in request.form
    vehicle_detection = 'vehicleDetection' in request.form
    smoking_detection = 'smokingDetection' in request.form
    coord1_x = request.form.get('coord1_x')
    coord1_y = request.form.get('coord1_y')
    coord2_x = request.form.get('coord2_x')
    coord2_y = request.form.get('coord2_y')
    coordinates = [coord1_x, coord1_y, coord2_x, coord2_y]

    cameras[cam_id] = {
        'indoor': indoor,
        'people_detection': people_detection,
        'fire_detection': fire_detection,
        'vehicle_detection': vehicle_detection,
        'smoking_detection': smoking_detection,
        'coordinates': coordinates
    }

    return render_template('index.html', cameras=cameras)
# Function to generate frames from camera


def generate_frames(cam_id):
    cap = cv2.VideoCapture(cam_id)  # OpenCV VideoCapture object for the camera
    while True:
        success, frame = cap.read()
        if not success:
            break


        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__=="__main__":
    app.run()