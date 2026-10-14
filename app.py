from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import os

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2   # 🔥 TWO HANDS (10 fingers)
)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(1)  # change to 0 if needed

finger_tips = [4, 8, 12, 16, 20]

# 🔥 Global real-time data
data = {
    "sign": "NO HAND",
    "count": 0,
    "status": "NOT DETECTED"
}

def generate_frames():
    global data

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        finger_count = 0
        status = "NOT DETECTED"
        sign = "NO HAND"

        if result.multi_hand_landmarks:
            status = "DETECTED"

            for handLms in result.multi_hand_landmarks:
                lm_list = []

                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))

                # Thumb
                if lm_list[4][1] > lm_list[3][1]:
                    finger_count += 1

                # Other fingers
                for tip in finger_tips[1:]:
                    if lm_list[tip][2] < lm_list[tip - 2][2]:
                        finger_count += 1

                mp_draw.draw_landmarks(
                    frame, handLms, mp_hands.HAND_CONNECTIONS
                )

        # ===== 10 SIGN MAPPING =====
        if finger_count == 0:
            sign = "HELLO"
        elif finger_count == 1:
            sign = "WHAT IS YOUR NAME?"
        elif finger_count == 2:
            sign = "MY NAME IS"
        elif finger_count == 3:
            sign = "HOW ARE YOU?"
        elif finger_count == 4:
            sign = "I AM FINE"
        elif finger_count == 5:
            sign = "THANK YOU"
        elif finger_count == 6:
            sign = "YES"
        elif finger_count == 7:
            sign = "NO"
        elif finger_count == 8:
            sign = "PLEASE"
        elif finger_count == 9:
            sign = "HELP"
        elif finger_count == 10:
            sign = "GOODBYE"
        else:
            sign = "UNKNOWN"

        # 🔥 Update live data
        data["sign"] = sign
        data["count"] = finger_count
        data["status"] = status

        cv2.putText(frame, sign, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def get_data():
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
