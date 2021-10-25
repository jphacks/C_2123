#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import os

from flask.scaffold import F

camera = cv2.VideoCapture(0)
# 分類器の指定
cascade_path = os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"
)
cascade = cv2.CascadeClassifier(cascade_path)

def gen_frames(): 

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # 画像の取得と顔の検出
            buffer = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            face_list = cascade.detectMultiScale(buffer, minSize=(100, 100))
            # 検出した顔に印を付ける
            for (x, y, w, h) in face_list:
                color = (0, 0, 225)
                pen_w = 3
                cv2.rectangle(buffer, (x, y), (x+w, y+h), color, thickness = pen_w)
            # 最後に、jpgとして保存し、byte列に変換する！
            ret, buffer = cv2.imencode('.jpg', buffer)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
#Initialize the Flask app


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)