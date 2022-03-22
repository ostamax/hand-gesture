import cv2


class VideoStreamer:
    """
    Video Streamer class
    """

    def __init__(self):
        # capture video from webcam - set 0
        self.cap = cv2.VideoCapture(0)
