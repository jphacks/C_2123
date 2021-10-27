# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
# 上記のページのサンプルコードを参考にしました

#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np

camera = cv2.VideoCapture(0)

#顔写真と、特徴量の登録
man1_image = face_recognition.load_image_file("man1-1.jpg")
man1_face_encoding = face_recognition.face_encodings(man1_image)[0]
man2_image = face_recognition.load_image_file("man2-1.jpg")
man2_face_encoding = face_recognition.face_encodings(man2_image)[0]
man3_image = face_recognition.load_image_file("man3-1.jpg")
man3_face_encoding = face_recognition.face_encodings(man3_image)[0]
man4_image = face_recognition.load_image_file("my-photo.jpg")
man4_face_encoding = face_recognition.face_encodings(man4_image)[0]

known_face_encodings = [
    man1_face_encoding,
    man2_face_encoding,
    man3_face_encoding,
    man4_face_encoding
]
known_face_names = [
    "man1",
    "man2",
    "man3",
    "myname"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            #画像のサイズを小さくして、顔処理の高速化、画像のカラーをRGBに変換(サイズを小さくしないと動画が重くなる)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                #動画内の顔を見つけて、特徴量化を行う
                frame_face_locations = face_recognition.face_locations(rgb_small_frame)
                frame_face_encodings = face_recognition.face_encodings(rgb_small_frame, frame_face_locations)
                frame_face_names = []

                for frame_face_encoding in frame_face_encodings:
                    # 登録してある顔と動画内の顔が一致するか確認する
                    matches = face_recognition.compare_faces(known_face_encodings, frame_face_encoding)
                    name = "Unknown"

                    # 動画内の顔ともっとも距離が近い顔を計算する
                    face_distances = face_recognition.face_distance(known_face_encodings, frame_face_encoding)
                    print(face_distances)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    frame_face_names.append(name)

            # 画面に表示する
            for (top, right, bottom, left), name in zip(frame_face_locations, frame_face_names):
                # 認識は大きさを小さくして行っていたため、大きさを元に戻す
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # 顔の周りに枠を作成する
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # 顔の下にラベルを作成する
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # 画像を表示する

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
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