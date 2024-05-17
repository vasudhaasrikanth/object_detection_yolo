import cv2
import numpy as np
import urllib.request

url = 'http://192.168.247.112/cam-hi.jpg'


# Load YOLO model
model_config = './yolov3.cfg'
model_weights = './yolov3.weights'
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Read class names
classes_file = './coco.names'
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Function to perform object detection
def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    
    h, w, _ = img.shape
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    
    if len(indices) > 0:  # Check if indices is not empty
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(img, f'{classes[class_ids[i]]} {confidences[i]:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# Main loop
while True:
    try:
        # Capture frame from ESP32-CAM
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        # Perform object detection
        detect_objects(frame)

        # Display the frame
        cv2.imshow('Object Detection', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Error:", e)
        break

# Release the camera and close OpenCV windows
cv2.destroyAllWindows()

