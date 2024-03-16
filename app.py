import cv2
from flask import Flask,render_template, request,redirect,Response



app = Flask(__name__)

cameras={}


@app.route('/')
def dashboard():
    return render_template('dashboard.html',cameras=cameras)


if __name__=="__main__":
    app.run()