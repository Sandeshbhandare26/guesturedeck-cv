
import cv2

def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Webcam could not be initialized.")
        return

    print("Camera initialized successfully!")
    print("Press Q to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)

            cv2.imshow("AirCommand - Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released successfully.")

if __name__ == "__main__":
    main()
