from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture("video.mp4")  
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
while True:
    success, frame = cap.read()
    if not success:
        break
    result = model(frame)[0]  
    boxes = result.boxes      
    for box in boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"{name} (Confidence: {confidence:.2f})")
    output_frame = result.plot()
    out.write(output_frame)
cap.release()
out.release()
print(" Detection done! Check 'output.mp4'")
