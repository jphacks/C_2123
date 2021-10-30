# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
# 上記のページのサンプルコードを参考にしました

#Import necessary libraries
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import numpy as np
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename



#顔写真と、特徴量の登録
man1_image = face_recognition.load_image_file("man1-1.jpg")
man1_face_encoding = face_recognition.face_encodings(man1_image)[0]
man2_image = face_recognition.load_image_file("man2-1.jpg")
man2_face_encoding = face_recognition.face_encodings(man2_image)[0]
man3_image = face_recognition.load_image_file("man3-1.jpg")
man3_face_encoding = face_recognition.face_encodings(man3_image)[0]
man4_image = face_recognition.load_image_file("my-photo.jpg")
man4_face_encoding = face_recognition.face_encodings(man4_image)[0]
'''
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
'''

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames(known_face_names, known_face_encodings,known_face_intros):
    camera = cv2.VideoCapture(0)  
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
                frame_face_intros = []

                for frame_face_encoding in frame_face_encodings:
                    # 登録してある顔と動画内の顔が一致するか確認する
                    matches = face_recognition.compare_faces(known_face_encodings, frame_face_encoding)
                    name = "Unknown"
                    # intro = "Unknown"

                    # 動画内の顔ともっとも距離が近い顔を計算する
                    face_distances = face_recognition.face_distance(known_face_encodings, frame_face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        intro = known_face_intros[best_match_index]

                    frame_face_names.append(name)
                    frame_face_intros.append(intro)

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
                cv2.rectangle(frame, (left, bottom - 70), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


            # 画像を表示する

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
#Initialize the Flask app

app = Flask(__name__)
# データベースの設定
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Test(db.Model):
    # カラムをここで定義
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(128), nullable=True)
    intro = db.Column(db.String(128), nullable=True)
    man_face_encoding_str = db.Column(db.String, nullable=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/registration')
def registration():
    # これでデータベースからデータを受け取れる
    data = Test.query.all()
    return render_template('registration.html', data=data)

@app.route('/video_feed')
def video_feed():
    data = Test.query.all()
    known_face_encodings = []
    known_face_names = []
    known_face_intros = []
    for d in data:
        known_face_names.append(d.name)
        known_face_intros.append(d.intro)
        known_face_encodings.append(list(map(float, d.man_face_encoding_str.split(","))))
    return Response(gen_frames(known_face_names, known_face_encodings,known_face_intros), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add', methods=['POST'])
def add():
    # クラスの定義
    new_name = Test()
    # 簡単に撮れるデータは、直接インスタンスに代入
    new_name.name = request.form["name"]
    new_name.intro = request.form['intro']
    # 画像ファイルは別途処理が必要
    image = request.files.get("image")
    filename = secure_filename(image.filename)
    filepath = "image/" + filename
    image.save(filepath)
    # 保存した画像の顔をencodingして配列に
    man_image = face_recognition.load_image_file(filepath)
    man_face_encoding = face_recognition.face_encodings(man_image)[0]
    # 配列はデータベースに入らないので、文字列に変換してインスタンスに渡す
    new_name.man_face_encoding_str = ','.join(str(x) for x in man_face_encoding)
    # データベースへ保存
    db.session.add(new_name)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)