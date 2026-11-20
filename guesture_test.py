
import cv2
import mediapipe as mp
import numpy as np
import time

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/hand_landmarker.task"
CAMERA_INDEX = 0
MAX_HANDS = 2

PINCH_THRESHOLD = 0.30

HISTORY_SIZE = 5
STABILITY_DURATION = 0.5
COOLDOWN_DURATION = 0.8

HAND_COLORS = {
    "Left": (0, 255, 0),
    "Right": (255, 0, 255)
}

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ==========================================
# DISTANCE CALCULATION
# ==========================================

def distance(p1, p2):
    return np.linalg.norm(
        np.array([p1.x, p1.y]) -
        np.array([p2.x, p2.y])
    )


# ==========================================
# FINGER EXTENSION DETECTION
# ==========================================

def get_finger_states(lm):

    wrist = lm[0]

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers = []

    for tip, pip in zip(finger_tips, finger_pips):

        tip_distance = distance(lm[tip], wrist)
        pip_distance = distance(lm[pip], wrist)

        extended = tip_distance > pip_distance * 1.12

        fingers.append(extended)

    # Thumb extension

    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    tip_distance = distance(thumb_tip, wrist)
    ip_distance = distance(thumb_ip, wrist)

    thumb_extended = (
        tip_distance > ip_distance * 1.12
        and distance(thumb_tip, lm[5])
        > distance(thumb_mcp, lm[5]) * 1.05
    )

    return thumb_extended, fingers


# ==========================================
# GESTURE RECOGNITION ENGINE
# ==========================================

def recognize_gesture(lm):

    thumb, fingers = get_finger_states(lm)

    index, middle, ring, pinky = fingers

    extended_count = sum(fingers)

    palm_scale = distance(lm[0], lm[9])

    if palm_scale < 0.001:
        return "UNKNOWN"

    pinch_distance = distance(lm[4], lm[8]) / palm_scale

    # 1. OK SIGN

    if pinch_distance < PINCH_THRESHOLD and middle and ring and pinky:
        return "OK SIGN"

    # 2. PINCH

    if pinch_distance < PINCH_THRESHOLD:
        return "PINCH"

    # 3. OPEN PALM

    if thumb and all(fingers):
        return "OPEN PALM"

    # 4. CLOSED FIST

    if not thumb and extended_count == 0:
        return "CLOSED FIST"

    # 5. THUMBS UP / DOWN

    if thumb and extended_count == 0:

        thumb_direction = lm[4].y - lm[2].y

        if thumb_direction < -0.05:
            return "THUMBS UP"

        elif thumb_direction > 0.05:
            return "THUMBS DOWN"

    # 6. PEACE SIGN

    if index and middle and not ring and not pinky:
        return "PEACE SIGN"

    # 7. POINTING UP

    if (
        index and not middle and not ring
        and not pinky and not thumb
    ):
        return "POINTING UP"

    # 8. THREE FINGERS

    if index and middle and ring and not pinky and not thumb:
        return "THREE FINGERS"

    # 9. FOUR FINGERS

    if all(fingers) and not thumb:
        return "FOUR FINGERS"

    # 10. ROCK SIGN

    if index and pinky and not middle and not ring:
        return "ROCK SIGN"

    # 11. SHAKA

    if thumb and pinky and not index and not middle and not ring:
        return "SHAKA"

    return "UNKNOWN"


# ==========================================
# GESTURE STABILIZATION ENGINE
# ==========================================

class GestureStabilizer:

    def __init__(self):

        self.history = deque(maxlen=HISTORY_SIZE)

        self.candidate = None
        self.candidate_start = 0

        self.confirmed_gesture = None

        self.last_event_time = 0

    def update(self, raw_gesture, current_time):

        self.history.append(raw_gesture)

        # Find the most frequently detected gesture
        counts = Counter(self.history)

        filtered_gesture, count = counts.most_common(1)[0]

        # Require a majority of recent frames
        if count < (HISTORY_SIZE // 2 + 1):

            return (
                self.confirmed_gesture or "NONE",
                "WAITING",
                None
            )

        # Ignore UNKNOWN gestures
        if filtered_gesture == "UNKNOWN":

            self.candidate = None

            return (
                self.confirmed_gesture or "NONE",
                "UNRECOGNIZED",
                None
            )

        # New candidate detected
        if filtered_gesture != self.candidate:

            self.candidate = filtered_gesture
            self.candidate_start = current_time

        # Already confirmed
        if filtered_gesture == self.confirmed_gesture:

            return (
                self.confirmed_gesture,
                "CONFIRMED",
                None
            )

        # Check stability duration
        stable_time = current_time - self.candidate_start

        if stable_time < STABILITY_DURATION:

            return (
                self.confirmed_gesture or "NONE",
                "STABILIZING",
                None
            )

        # Check cooldown
        if current_time - self.last_event_time < COOLDOWN_DURATION:

            return (
                self.confirmed_gesture or "NONE",
                "COOLDOWN",
                None
            )

        # Confirm new gesture
        self.confirmed_gesture = filtered_gesture
        self.last_event_time = current_time

        # Emit one event
        return (
            self.confirmed_gesture,
            "CONFIRMED",
            self.confirmed_gesture
        )

    def reset(self):

        self.history.clear()

        self.candidate = None
        self.candidate_start = 0

        self.confirmed_gesture = None


# ==========================================
# DRAW HAND LANDMARKS
# ==========================================

def draw_hand(frame, landmarks, color):

    height, width, _ = frame.shape

    for point in landmarks:

        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(frame, (x, y), 5, color, -1)

    for start, end in CONNECTIONS:

        p1 = landmarks[start]
        p2 = landmarks[end]

        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)

        cv2.line(frame, (x1, y1), (x2, y2), color, 2)


# ==========================================
# DRAW STATUS PANEL
# ==========================================

def draw_status_panel(frame, hand_states, last_event):

    cv2.rectangle(
        frame,
        (10, 10),
        (790, 255),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "AIRCOMMAND - GESTURE STABILIZATION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    for i, hand_name in enumerate(["Left", "Right"]):

        state = hand_states[hand_name]

        color = HAND_COLORS[hand_name]

        y = 90 + i * 75

        if state["detected"]:

            cv2.putText(
                frame,
                f"{hand_name} Hand",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Raw: {state['raw']}",
                (200, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                2
            )

            cv2.putText(
                frame,
                f"Stable: {state['stable']}",
                (200, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status: {state['status']}",
                (500, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                f"{hand_name} Hand: NOT DETECTED",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.putText(
        frame,
        f"Last Event: {last_event}",
        (20, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    detector = None
    camera = None

    stabilizers = {
        "Left": GestureStabilizer(),
        "Right": GestureStabilizer()
    }

    last_event = "NONE"

    try:

        # Initialize MediaPipe

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        detector = vision.HandLandmarker.create_from_options(options)

        # Initialize webcam

        camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():
            raise RuntimeError("Webcam initialization failed.")

        print("=" * 50)
        print("AIRCOMMAND - STABILIZED GESTURE RECOGNITION")
        print("=" * 50)
        print("Maximum hands:", MAX_HANDS)
        print("Stability duration:", STABILITY_DURATION, "seconds")
        print("Cooldown:", COOLDOWN_DURATION, "seconds")
        print("Press Q to exit.")
        print("=" * 50)

        while True:

            success, frame = camera.read()

            if not success:
                print("Camera frame unavailable.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame)
            )

            result = detector.detect(mp_image)

            current_time = time.monotonic()

            hand_states = {

                "Left": {
                    "detected": False,
                    "raw": "NONE",
                    "stable": "NONE",
                    "status": "NOT DETECTED"
                },

                "Right": {
                    "detected": False,
                    "raw": "NONE",
                    "stable": "NONE",
                    "status": "NOT DETECTED"
                }
            }

            detected_hands = set()

            if result.hand_landmarks:

                for hand_index, landmarks in enumerate(
                    result.hand_landmarks
                ):

                    handedness = result.handedness[hand_index][0]

                    hand_name = handedness.category_name

                    if hand_name not in hand_states:
                        continue

                    detected_hands.add(hand_name)

                    # Raw gesture recognition
                    raw_gesture = recognize_gesture(landmarks)

                    # Stabilization
                    stable_gesture, status, event = (
                        stabilizers[hand_name].update(
                            raw_gesture,
                            current_time
                        )
                    )

                    hand_states[hand_name] = {
                        "detected": True,
                        "raw": raw_gesture,
                        "stable": stable_gesture,
                        "status": status
                    }

                    # New confirmed gesture event
                    if event is not None:

                        last_event = f"{hand_name} -> {event}"

                        print(
                            f"[GESTURE EVENT] {last_event}"
                        )

                    # Draw hand
                    draw_hand(
                        frame,
                        landmarks,
                        HAND_COLORS[hand_name]
                    )

            # Reset a hand if it disappears
            for hand_name in ["Left", "Right"]:

                if hand_name not in detected_hands:

                    stabilizers[hand_name].reset()

            # Draw interface
            draw_status_panel(
                frame,
                hand_states,
                last_event
            )

            hand_count = len(detected_hands)

            cv2.putText(
                frame,
                f"Hands Detected: {hand_count}/{MAX_HANDS}",
                (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "AirCommand - Stabilized Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:

        print("ERROR:", error)

    finally:

        if camera is not None:
            camera.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()

        print("AirCommand closed successfully.")


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
