from ultralytics import YOLO
import cv2
model = YOLO("yolov8x.pt")
image_file = "Golden-Retriever.webp"
results = model(image_file)
results[0].save(filename="output.jpg")
print("Detection complete! Check 'output.jpg'")
for result in results:
    boxes = result.boxes
    print("Objects found:")
    for box in boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"- {name} (Confidence: {confidence:.2f})")
output_image = cv2.imread("output.jpg")  


