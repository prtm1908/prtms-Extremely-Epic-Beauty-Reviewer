from flask import Flask, render_template, Response, url_for
import threading

import cv2
from deepface import DeepFace
app=Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# counter = 0

reference_img = cv2.imread("Person/Pratham.jpg")

face_match = False

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False
        
def gen_frames():  
    counter=0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Hi")
            break
        else:
            if counter<200:
                cv2.putText(frame, "Analyzing...", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            if counter % 30 == 0:
                try:
                    threading.Thread(target=check_face, args=(frame.copy(),)).start()
                except ValueError:
                    pass
            counter += 1
            if face_match and counter>=200:
                cv2.putText(frame, "11/10 beauty (>\\\\<)", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if not face_match and counter>=200:
                cv2.putText(frame, "Dev's dream girl not detected, ur ugly :(", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            success, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run()
