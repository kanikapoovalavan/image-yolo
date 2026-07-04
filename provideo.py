from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open input video
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video
out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Run YOLO detection
    results = model(frame)[0]

    # Print detected objects
    for box in results.boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"{name} (Confidence: {confidence:.2f})")

    # Draw detections
    output_frame = results.plot()

    # Save frame to output video
    out.write(output_frame)

    # Display video (optional)
    cv2.imshow("YOLOv8 Video Detection", output_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Detection completed!")
print("Output video saved as 'output.mp4'")
