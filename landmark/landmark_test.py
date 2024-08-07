import os, sys
sys.path.append(os.path.abspath('.'))

from dataset import Dataset
from holistic_landmark import HolisticModel
import cv2

dataset = Dataset()
sample_data = dataset.load(size=1)
# print(sample_data)

# Init Holistic model
holistic = HolisticModel()

def preprocessing(features):
    pass

for frame in sample_data[0]["frames"]:
    features, image = holistic.get_landmarks(frame)
    print(features)

    preprocessing(features)

    cv2.imshow("test", image)
    cv2.waitKey(0)
    
