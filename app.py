from flask import Flask, render_template, request, Response
from detector import detect_vehicles, live_detection, model
from logic import get_signal_time
import cv2
import time
import random
import numpy as np 

vehicle_tracks = {}
vehicle_id = 0
traffic_history = []

app = Flask(__name__)

# 🔥 LANE DETECTION
def detect_lanes(frame):
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (0, int(height*0.6))
    ]])

    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180,
        threshold=100,
        minLineLength=150,
        maxLineGap=50
    )

    left, right = [], []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]

            if x2-x1 == 0:
                continue

            slope = (y2-y1)/(x2-x1)

            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left.append(line[0])
            else:
                right.append(line[0])

    def avg_line(lines):
        if len(lines) == 0:
            return None
        return np.mean(lines, axis=0).astype(int)

    left_line = avg_line(left)
    right_line = avg_line(right)

    lane_center = None

    if left_line is not None:
        x1,y1,x2,y2 = left_line
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,255),5)

    if right_line is not None:
        x1,y1,x2,y2 = right_line
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,255),5)

    if left_line is not None and right_line is not None:
        lx1,ly1,lx2,ly2 = left_line
        rx1,ry1,rx2,ry2 = right_line

        pts = np.array([
            [lx1,ly1],
            [rx1,ry1],
            [rx2,ry2],
            [lx2,ly2]
        ])

        cv2.fillPoly(frame,[pts],(0,255,0))
        lane_center = int((lx1 + rx1)/2)

        cv2.line(frame,(lane_center,height),
                 (lane_center,int(height*0.6)),
                 (255,255,0),2)

    return frame, lane_center


# 🎥 LIVE STREAM
def generate_frames():
    global vehicle_id

    cap = cv2.VideoCapture("sample.mp4")

    if not cap.isOpened():
        print("❌ Video not found")
        return

    blink = True
    last_toggle = time.time()

    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))

        # 🔥 LANE DETECTION
        frame, lane_center = detect_lanes(frame)

        results = model(frame, verbose=False)

        count = 0

        # 🔥 FIXED TRACKING LOOP
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in [2,3,5,7]:

                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2)/2)
                    cy = int((y1 + y2)/2)

                    matched_id = None

                    # 🔥 FIXED UNPACKING HERE
                    for vid, (px, py, _, _, _, _) in vehicle_tracks.items():
                        if abs(cx - px) < 50 and abs(cy - py) < 50:
                            matched_id = vid
                            break

                    if matched_id is None:
                        vehicle_id += 1
                        matched_id = vehicle_id

                    vehicle_tracks[matched_id] = (cx,cy,x1,y1,x2,y2)

                    count += 1

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(frame, f"ID:{matched_id}",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),2)

        # 💥 ACCURATE COLLISION DETECTION (FIXED INDENTATION)
        vehicles = list(vehicle_tracks.values())

        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):

                _, _, x1a, y1a, x2a, y2a = vehicles[i]
                _, _, x1b, y1b, x2b, y2b = vehicles[j]

                xa = max(x1a, x1b)
                ya = max(y1a, y1b)
                xb = min(x2a, x2b)
                yb = min(y2a, y2b)

                overlap = max(0, xb - xa) * max(0, yb - ya)

                cx1 = (x1a + x2a) // 2
                cy1 = (y1a + y2a) // 2
                cx2 = (x1b + x2b) // 2
                cy2 = (y1b + y2b) // 2

                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

                if overlap > 0 or dist < 40:
                    cv2.putText(frame, "COLLISION WARNING",
                                (200,250),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,(0,0,255),2)

        # 📊 TRAFFIC PREDICTION
        traffic_history.append(count)

        if len(traffic_history) > 20:
            traffic_history.pop(0)

        if len(traffic_history) >= 5:
            trend = sum(traffic_history[-5:]) / 5

            if trend > count:
                cv2.putText(frame, "TRAFFIC INCREASING",
                            (20,210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(255,255,0),2)
            else:
                cv2.putText(frame, "TRAFFIC STABLE",
                            (20,210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,255),2)

        # 🚗 LANE LOGIC
        frame_center = 320

        if lane_center is not None:
            deviation = frame_center - lane_center

            if abs(deviation) > 40:
                cv2.putText(frame, "LANE DEPARTURE",
                            (200,200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,0,255),2)

            if abs(deviation) < 40:
                cv2.putText(frame, "LANE: SAFE",
                            (20,180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,0),2)
            else:
                cv2.putText(frame, "LANE: WARNING",
                            (20,180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,0,255),2)

        # 🔥 BLINK BORDER
        if time.time() - last_toggle > 0.5:
            blink = not blink
            last_toggle = time.time()

        color = (0,255,0) if blink else (0,150,0)
        cv2.rectangle(frame, (5,5), (635,475), color, 2)

        # 🔢 VEHICLE COUNT
        cv2.putText(frame, f"Vehicles: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 🧠 HUD
        cv2.putText(frame, "AI TRAFFIC MONITOR", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, "SIMULATION MODE", (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        speed = random.randint(20,80)
        cv2.putText(frame, f"Speed: {speed} km/h", (20,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

        if count > 15:
            cv2.putText(frame, "HIGH TRAFFIC", (350,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    priority = []

    if request.method == "POST":
        road_count = int(request.form.get("roadCount", 0))

        for i in range(1, road_count + 1):
            file = request.files.get(f"video_{i}")
            name = request.form.get(f"name_{i}")

            if not file or not name:
                continue

            path = f"temp_{i}.mp4"
            file.save(path)

            count, ambulance = detect_vehicles(path)

            if ambulance:
                time = 60
                density = "Emergency 🚑"
            else:
                time, density = get_signal_time(count)

            results.append({
                "name": name,
                "count": count,
                "time": time,
                "density": density
            })

        priority = sorted(results, key=lambda x: x["count"], reverse=True)

    return render_template("index.html",
                           results=results,
                           priority=priority)


if __name__ == "__main__":
    app.run(debug=True)