import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = r"C:/Users/Hxtreme/Videos/record.mp4"  # <-- Your local path here
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
