import cv2
import numpy as np
import plotly.express as px
import pytesseract as pt

# settings
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('C:\Users\Technical Support\Downloads\ALPR Test Image 1\ALPR Test Image\yolov5\runs\train\Model\weights\best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Hàm lấy dự đoán từ YOLO
def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Dự đoán từ YOLO
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

# Hàm non-maximum suppression
def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

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

# Vẽ bounding box và biển số xe
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

# Hàm lấy biển số xe từ ảnh
def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    if 0 in roi.shape:
        return 'no number'
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        return text

# Dự đoán YOLO cho ảnh
def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img = drawings(img, boxes_np, confidences_np, index)
    return result_img

# Hàm xử lý video
def process_video(video_path, net):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image, detections = get_detections(frame, net)
        boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
        result_img = drawings(frame, boxes_np, confidences_np, index)

        # Hiển thị khung hình
        cv2.imshow('YOLO Video', result_img)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm xử lý ảnh
def process_image(image_path, net):
    img = cv2.imread(image_path)
    result_img = yolo_predictions(img, net)

    # Hiển thị kết quả ảnh
    fig = px.imshow(result_img)
    fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()

# Lựa chọn input (ảnh hay video)
input_type = 'image'  # hoặc 'video'

if input_type == 'image':
    # Xử lý ảnh
    process_image('C:\Users\Technical Support\Downloads\ALPR Test Image 1\ALPR Test Image\yolov5\data\images\val\H402.jpg', net)
elif input_type == 'video':
    # Xử lý video
    process_video('path_to_video.mp4', net)
