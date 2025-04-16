from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pytesseract
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

# Tesseractのパス（Windows環境用）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 学習済みモデルの読み込み
model = load_model("modelsave2.h5")

# ラベル→文字マッピング（ -, 0-9, 「, a-z, 」, A-Z ）
label_to_char = {}
label_to_char[0] = '-'
for i in range(10):
    label_to_char[i + 1] = str(i)
label_to_char[11] = '「'
for i in range(26):
    label_to_char[i + 12] = chr(ord('a') + i)
label_to_char[38] = '」'
for i in range(26):
    label_to_char[i + 39] = chr(ord('A') + i)

# 前処理関数
def preprocess_image_from_array(img_array, target_size=(128, 128)):
    img_resized = cv2.resize(img_array, target_size)
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 外部から画像を読み込み
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    # 前処理：リサイズ + グレースケール + 二値化
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 文字領域検出（Tesseractのboxes使用）
    h_img, w_img = thresh.shape
    boxes = pytesseract.image_to_boxes(thresh)

    # 各文字画像の切り出し + x位置でソート
    char_images = []
    for b in boxes.splitlines():
        b = b.split()
        if len(b) != 6:
            continue
        _, x1, y1, x2, y2 = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])
        y1 = h_img - y1
        y2 = h_img - y2
        char_img = image[y2:y1, x1:x2]
        char_images.append((x1, char_img))  # x1 でソート

    char_images.sort(key=lambda x: x[0])  # 左→右順に

    # 推論と文字列化
    predicted_text = ""
    for _, char_img in char_images:
        inverted = cv2.bitwise_not(char_img)
        preprocessed = preprocess_image_from_array(inverted)
        prediction = model.predict(preprocessed)
        label = int(np.argmax(prediction))
        predicted_char = label_to_char.get(label, "?")
        predicted_text += predicted_char

    return JSONResponse(content={"result": predicted_text})
