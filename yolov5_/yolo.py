import cv2
import numpy as np
import onnxruntime as ort
import pytesseract as pt

# Các tham số
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
MODEL_PATH = 'C:/Users/Technical Support/Downloads/ALPR Test Image 1/ALPR Test Image/yolov5/yolov5s.onnx'
IMAGE_PATH = 'C:/Users/Technical Support/Downloads/ALPR Test Image 1/ALPR Test Image/H502.jpg'

# Đọc ảnh
img = cv2.imread(IMAGE_PATH)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img)
img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
# Tải mô hình sử dụng onnxruntime
session = ort.InferenceSession(MODEL_PATH)

# Thực hiện dự đoán mẫu
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Hàm lấy phát hiện từ mô hình YOLO
def get_detections(img, session):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Chuyển đổi ảnh thành định dạng YOLO
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    input_data = blob.astype(np.float32)

    # Dự đoán
    output = session.run([output_name], {input_name: input_data})[0]
    detections = output[0]
    
    # In ra kết quả để kiểm tra
    print("Detections:", detections)
    
    return input_image, detections

# Hàm xử lý NMS (Non-Maximum Suppression) để lọc các kết quả dự đoán
def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / 640
    y_factor = image_h / 640

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index

# Hàm vẽ hộp và ký tự lên ảnh
def drawings(image, boxes_np, confidences_np, index):
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)

        license_text = extract_text(image, boxes_np[ind])

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return image

# Hàm trích xuất văn bản (số biển số)
def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    if 0 in roi.shape:
        return 'no number'
    else:
        text = pt.image_to_string(roi)
        text = text.strip()

        return text

# Hàm chạy dự đoán trên ảnh
def yolo_predictions(img, session):
    input_image, detections = get_detections(img, session)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img = drawings(img, boxes_np, confidences_np, index)
    return result_img

# Chạy dự đoán trên ảnh đã tải
results = yolo_predictions(img, session)

# Hiển thị kết quả
import plotly.express as px

fig = px.imshow(results)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.show()