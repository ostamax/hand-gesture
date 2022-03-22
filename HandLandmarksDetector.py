import cv2
import mediapipe as mp


class HandLandmarksDetector:
    """
    Class that implements hand landmark's detection.
    """

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    def detector(self, frame):
        """
        Detects hand's landmarks on the frame.

        :param frame: an input image from web-camera.

        :return: a list of landmark's coordinates.
        """

        x, y, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        landmarks = list()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

        return landmarks
