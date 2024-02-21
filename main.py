import pandas as pd
import numpy as np
import cv2

from modules.get_landmarks import getLandmarks

def input():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring camera frame")
            continue
        
        getLandmarks(image)

    cap.release()
input()