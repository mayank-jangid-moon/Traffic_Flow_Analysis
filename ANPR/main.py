from ultralytics import YOLO
from sort.sort import *
import cv2
import util
from util import corr_car,write_csv,read_plate

tracker=Sort()
model=YOLO("yolov8x_best.pt")
anpd=YOLO("./models/license_plate_detector.pt")

cap=cv2.VideoCapture('2103099-uhd_3840_2160_30fps.mp4')
results={}
ret=True
count=-1
while ret:
    count+=1
    
    ret,frame=cap.read()
    if ret :
        results[count]={}
        detections=model(frame)[0]
        detection_frame=[]
        for detect in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls=detect
            detection_frame.append([x1,y1,x2,y2,conf])
        
        track_id=tracker.update(np.asarray(detection_frame))
        number_plates=anpd(frame)[0]
        for numberplate in number_plates.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls=numberplate
            x1car,y1car,x2car,y2car,carid=corr_car(numberplate,track_id)
            if carid!=-1:
                crop=frame[int(y1):int(y2),int(x1):int(x2),:]
                gray_crop=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                _,inverted_gray_crop=cv2.threshold(gray_crop,64,255,cv2.THRESH_BINARY_INV)
                plate_text,plate_score=read_plate(inverted_gray_crop)
            
                if plate_text is not None:
                    results[count][carid] = {'car': {'bbox': [x1car, y1car, x2car, y2car]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': plate_text,
                                                                    'bbox_score': cls,
                                                                    'text_score': plate_score}}
                
write_csv(results, './test.csv')