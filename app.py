import cv2
from flask import Flask,render_template, request,redirect,Response



app = Flask(__name__)

cameras={}
