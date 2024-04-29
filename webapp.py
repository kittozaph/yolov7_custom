"""
View the inference results on the image in the browser.
"""
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('test.html')


# function for accessing rtsp stream 1
#cap1 = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')

@app.route("/rtsp_feed1")
def rtsp_feed1():
    def gen_frames():
        while True:
            # Read a frame from the webcam
            success, frame = cap3.read()
            if not success:
                break
            else:
                # Encode the frame as a JPEG image
                ret, buffer = cv2.imencode('.jpg', frame)
                # Convert the image to a byte array
                frame = buffer.tobytes()
                # Yield the frame as a multipart response
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)
    # Return the webcam feed as a response
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# function for accessing rtsp stream 2
#cap2 = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')

@app.route("/rtsp_feed2")
def rtsp_feed2():
    def gen_frames():
        while True:
            # Read a frame from the webcam
            success, frame = cap3.read()
            if not success:
                break
            else:
                # Encode the frame as a JPEG image
                ret, buffer = cv2.imencode('.jpg', frame)
                # Convert the image to a byte array
                frame = buffer.tobytes()
                # Yield the frame as a multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)   
    # Return the webcam feed as a response
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to start webcam and detect objects
# Open the webcam
cap3 = cv2.VideoCapture(0)

#@app.route('/webcam_feed')
#def webcam_feed():
    # Function to generate the webcam frames
    #def gen_frames():
        #while True:
            # Read a frame from the webcam
            #success, frame = cap3.read()
            #if not success:
                #break
            #else:
                # Encode the frame as a JPEG image
                #ret, buffer = cv2.imencode('.jpg', frame)
                # Convert the image to a byte array
                #frame = buffer.tobytes()
                # Yield the frame as a multipart response
                #yield (b'--frame\r\n'
                       #b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # Return the webcam feed as a response
    #return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('test.html')

    else:
        return "Invalid file format"

    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()    
            if file_extension == 'jpg':
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","crayfish.pt"], shell=True)
                process.wait()
                
                
            elif file_extension == 'mp4':
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","crayfish.pt"], shell=True)
                process.communicate()
                process.wait()

            
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('test.html', image_path=image_path)
    #return "done"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','crayfish.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

