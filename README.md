# Who are you?
[動画のリンク先](https://drive.google.com/drive/folders/12sfFwGB98a76Ub2LPS8PLfxKcxiJzxH8?usp=sharing)
[![IMAGE ALT TEXT HERE](https://jphacks.com/wp-content/uploads/2021/07/JPHACKS2021_ogp.jpg)](https://www.youtube.com/watch?v=LUPQFB4QyVo)

## 製品概要
### 背景(製品開発のきっかけ、課題等）
- 大人数での交流会や歓迎会などで、周りの人の名前を知らなかったり、共通の話題が見つからなくて話しかけることに躊躇する。一回会ったことはあるけれど名前を思い出すことができない。そのような経験がある人も多いと思います。
- コロナも収束に向かっていく中、対面でのコミュニケーションが増えていくと思います。オンラインだと小さく名前が表示されていますが、対面だとそんな機能はありません。
- そこで、対面であっても相手の基本情報を知ることができれば、話しかけるハードルもかなり下がるのではないかと考えました。


### 製品説明（具体的な製品の説明）
- 事前に、顔写真と自分の表示したい名前、自己紹介を登録します
- その後、顔認証のページに入るとリアルタイムにカメラの映像を読み込み顔の周りに枠と名前、自己紹介が表示されます
### 特長
#### 1. 特長1
- 顔画像を登録できる
#### 2. 特長2
- 顔認証により、登録した情報をリアルタイムの映像に表示できる

### 解決出来ること
- コミュニケーションコストを下げる！
### 今後の展望
- コミュニティごとに情報を登録できるようにする
- スマホでカメラの起動/情報の登録をできるようにする
- ARでの顔認証とタグの表示をできるようにする
- カメラロールの静止画など、リアルタイムの映像でないものから顔認証を行い情報を表示できるようにする
### 注力したこと（こだわり等）
* 顔画像を登録する際、画像そのものではなく、特徴量化したものを保存することで、データベースに保存出来るようにした

## 開発技術
### 活用した技術
#### フレームワーク・ライブラリ・モジュール
* flask
* cv2
* face_recognition
* numpy
* flask_sqlalchemy
* werkzeug.utils

#### デバイス
* pc

### 独自技術
#### ハッカソンで開発した独自機能・技術
* 一週間でアイディアから実装まで全て行いました

#### 製品に取り入れた研究内容（データ・ソフトウェアなど）（※アカデミック部門の場合のみ提出必須）
* face_recognition