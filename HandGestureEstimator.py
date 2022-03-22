import numpy as np
from tensorflow.keras.models import load_model


class HandGestureEstimator:
    """
    Class that implements hand gesture estimation.
    """

    def __init__(self):
        # model and classes loading
        self.model = load_model('mp_hand_gesture')
        f = open('gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()

    def estimate(self, landmarks):
        """
        Estimates hand's gesture using previously detected landmarks.

        :param landmarks: list of detected hand's landmarks.
        
        :return: string, a name of a hand gesture.
        """

        if len(landmarks) > 0:
            prediction = self.model.predict([landmarks])
            classID = np.argmax(prediction)
            className = self.classNames[classID]

            return className

