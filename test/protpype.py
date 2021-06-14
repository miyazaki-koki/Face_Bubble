import cv2

from f_d import inference_usbCam_face

cascade_path = "haarcascade_frontalface_default.xml"

image_path = "test2.jpg"

color = (255, 255, 255) #白
#color = (0, 0, 0) #黒

inference_usbCam_face.Main(image_path)

"""
#ファイル読み込み
image = cv2.imread(image_path)

#グレースケール変換
image_gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)
#facerect = cascade.detectMultiScale(image_gray)
#facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

print("face rectangle")
print(facerect)

if len(facerect) > 0:
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=10)
"""

#認識結果の保存
cv2.imwrite("detected.jpg", image)
