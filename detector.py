from ultralytics import YOLO
import cv2

# 🔥 Load model once (important for performance)
model = YOLO("yolov8n.pt")

def detect_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)

    total = 0
    frames = 0
    ambulance_detected = False

    frame_skip = 5   # ⚡ speed optimization
    frame_count = 0
    max_frames = 100  # ⚡ prevent long processing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ⚡ Skip frames for speed
        if frame_count % frame_skip != 0:
            continue

        # ⚡ Resize frame (faster detection)
        frame = cv2.resize(frame, (640, 480))

        results = model(frame, verbose=False)

        count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # 🚗 vehicle classes
                if cls in [2, 3, 5, 7]:
                    count += 1

                # 🚑 ambulance heuristic
                if cls == 2 and conf > 0.8:
                    ambulance_detected = True

        total += count
        frames += 1

        # ⚡ limit processing
        if frames > max_frames:
            break

    cap.release()

    if frames == 0:
        return 0, False

    avg_count = total // frames

    return avg_count, ambulance_detected


# 🎥 Optional OpenCV live window
def live_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, verbose=False)

        count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2,3,5,7]:
                    count += 1

        cv2.putText(frame, f"Vehicles: {count}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()