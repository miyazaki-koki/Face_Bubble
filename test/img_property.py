import cv2
import sys

if __name__ == "__main__":
    img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)

    # 画像ファイルの読み込みに失敗したらエラー終了
    if img is None:
        print("Failed to load image file.")
        sys.exit(1)

    # カラーとグレースケールで場合分け
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
        channels = 1

    # 取得結果（幅，高さ，チャンネル数，depth）を表示
    print("width: " + str(width))
    print("height: " + str(height))
    print("channels: " + str(channels))
    print("dtype: " + str(img.dtype))
