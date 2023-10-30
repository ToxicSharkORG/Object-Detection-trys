from ultralytics import YOLO
import cv2, math, time, pyttsx3

cap = cv2.VideoCapture(0)

# model
model = YOLO("yolov8n.pt")
engine = pyttsx3.Engine()

cap.set(3, 640)
cap.set(4, 480)

# specified classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    print('running')
    success, img = cap.read()
    results = model(img, stream=True)

    # coords
    for r in results:
        boxes = r.boxes

        for box in boxes:
            #boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # box in camera screen broooo
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # condifence
            confidence = math.ceil((box.conf[0]*100))/100

            #predicted classes
            cls = int(box.cls[0])

            # location in screen
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, f"Confidence: {confidence}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print("Confidence --->",confidence)
            print("Class name -->", classNames[cls])
            engine.say(f'There is a {classNames[cls]} in front of you!, I am {int(confidence*100)}%s sure!')
            engine.runAndWait()
            
        cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()