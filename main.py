import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(1)  # change to 0 if needed

finger_tips = [4, 8, 12, 16, 20]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ===== UI CANVAS =====
    ui = np.zeros((h, w + 400, 3), dtype=np.uint8)
    ui[:] = (20, 20, 20)

    # Camera feed
    ui[0:h, 0:w] = frame

    # Right panel
    cv2.rectangle(ui, (w, 0), (w + 400, h), (35, 35, 35), -1)

    # Title
    cv2.putText(ui, "SIGN LANGUAGE SYSTEM",
                (w + 30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0
    text = "NO HAND"
    status = "NOT DETECTED"

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
                ui[:, :w], handLms, mp_hands.HAND_CONNECTIONS)

    # ===== 10 SIGN MAPPING =====
    if finger_count == 0:
        text = "HELLO"
    elif finger_count == 1:
        text = "WHAT IS YOUR NAME?"
    elif finger_count == 2:
        text = "MY NAME IS"
    elif finger_count == 3:
        text = "HOW ARE YOU?"
    elif finger_count == 4:
        text = "I AM FINE"
    elif finger_count == 5:
        text = "THANK YOU"
    elif finger_count == 6:
        text = "YES"
    elif finger_count == 7:
        text = "NO"
    elif finger_count == 8:
        text = "PLEASE"
    elif finger_count == 9:
        text = "HELP"
    elif finger_count == 10:
        text = "GOODBYE"

    # ===== INFO PANELS =====
    cv2.rectangle(ui, (w + 20, 80), (w + 380, 170), (0, 0, 0), 2)
    cv2.putText(ui, "DETECTED SIGN",
                (w + 80, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(ui, text,
                (w + 40, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.rectangle(ui, (w + 20, 190), (w + 380, 260), (0, 0, 0), 2)
    cv2.putText(ui, f"Finger Count : {finger_count}",
                (w + 50, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    cv2.rectangle(ui, (w + 20, 280), (w + 380, 350), (0, 0, 0), 2)
    cv2.putText(ui, f"Status : {status}",
                (w + 70, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 200, 255), 2)

    # Footer
    cv2.rectangle(ui, (0, h - 40), (w + 400, h), (0, 0, 0), -1)
    cv2.putText(ui, "Press Q to Exit",
                (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("Sign Language Detection System", ui)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
