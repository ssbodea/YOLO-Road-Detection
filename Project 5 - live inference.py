import cv2
from ultralytics import YOLO
import yt_dlp

YOUTUBE_URL = "https://www.youtube.com/watch?v=h1wly909BYw"
MODEL_PATH = r"D:\SBC_PROIECT\trained_models\yolov8n\weights\best.pt"
IMG_SIZE = 768  # You can reduce to 320 for faster FPS

# Get direct video URL from YouTube
def get_stream_url(url):
    with yt_dlp.YoutubeDL({"format": "best"}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]

stream_url = get_stream_url(YOUTUBE_URL)

# Load YOLO model once
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError("Cannot open livestream!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model.predict(frame, imgsz=IMG_SIZE, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Livestream", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
