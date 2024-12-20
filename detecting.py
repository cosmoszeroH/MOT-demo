import numpy as np
import cv2 as cv
import wget
import os


def convert_center2tlbr(centerX, centerY, width, height):
    """
    Convert bounding box of center, width, height format to top-left, bottom-right format
    """
    x1 = int(centerX - (width / 2))
    y1 = int(centerY - (height / 2))
    x2 = int(centerX + (width / 2))
    y2 = int(centerY + (height / 2))

    return [x1, y1, x2, y2]


def initialize_detection_model():
    cfg_file = '.\darknet\cfg\yolov3.cfg'
    weight_path = 'https://github.com/hank-ai/darknet/releases/download/v2.0/yolov3.weights'
    if not os.path.exists('yolov3.weights'):
        weight_file = wget.download(weight_path)
    else:
        weight_file = 'yolov3.weights'

    net = cv.dnn.readNet(config=cfg_file, model=weight_file)

    return net


def detect_object_in_image(model, image):
    h, w = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    model.setInput(blob)
    ln = model.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    except IndexError:
        ln = [ln[i - 1] for i in model.getUnconnectedOutLayers()]

    outputs = model.forward(ln)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                box = convert_center2tlbr(centerX, centerY, width, height)
                boxes.append(box)
                confidences.append(float(confidence))
                class_ids.append(classID)

    not_filtered_indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for index in range(len(class_ids)-1, -1, -1):
        if index not in not_filtered_indices:
            boxes.pop(index)
            confidences.pop(index)
            class_ids.pop(index)

    return boxes, confidences, class_ids
        

if __name__ == "__main__":
    model = initialize_detection_model()
    image = cv.imread(r'.\Screenshot.png')
    a, b, c = detect_object_in_image(model, image)
    print(a)