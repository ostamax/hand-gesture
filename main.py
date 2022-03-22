import cv2
import time
from VideoStreamer import VideoStreamer
from HandLandmarksDetector import HandLandmarksDetector
from HandGestureEstimator import HandGestureEstimator

vs = VideoStreamer()
hld = HandLandmarksDetector()
hge = HandGestureEstimator()

time.sleep(2)

while True:
    ret, frame = vs.cap.read()

    # mirror the current frame
    frame = cv2.flip(frame, 1)
    start = time.time()
    landmarks = hld.detector(frame)
    print('land', time.time() - start)
    start = time.time()
    className = hge.estimate(landmarks)
    print('pose', time.time() - start)

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # display
    cv2.imshow('GestureRecognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.cap.release()
cv2.destroyAllWindows()