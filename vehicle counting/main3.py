import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cvzone
from tracker import Tracker
import os
from datetime import datetime

# Load the model and class list
model = YOLO("yolov10s.pt")
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Initialize the tracker
tracker = Tracker()

# Set the line position and offset
cy1 = 425
offset = 6

# Initialize variables

listcardown = []
count = 0

# Create directory to save images if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cars.mp4')

while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
       break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
           list.append([x1, y1, x2, y2])
    
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if cy1<(cy+offset) and cy1>(cy-offset): 
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            if listcardown.count(id)==0:
                listcardown.append(id)
                # Crop and save the car image
                car_image = frame[y3:y4, x3:x4]
                # Resize the cropped image to a larger size
                resized_car_image = cv2.resize(car_image, (300, 300))  # Resize to 300x300 pixels or any desired size
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                cv2.imwrite(f"images/car_{timestamp}.jpg", resized_car_image)
                 
    cv2.line(frame, (343, 425), (961, 425), (255, 255, 255), 1)
    cardown = len(listcardown)
    cvzone.putTextRect(frame, f'Cardown:-{cardown}', (50, 60), 2, 2)
    
    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
