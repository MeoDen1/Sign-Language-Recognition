import pickle
import numpy as np
import gzip
import random
import os
import cv2

class Dataset:
    def __init__(self) -> None:
        with gzip.open('Dataset/phoenix14t.pami0.train.annotations_only.gzip', 'rb') as f:
            # annotations: list of object
            # - name: path (train/...)
            # - signer: signer name
            # - gloss: JETZT ...
            # - text: ...
            self.annotations = pickle.load(f)

    def load(self, path="Dataset/videos_phoenix/videos", size: int=10, pre_processing=None) -> list:
        """
        Return format: List of object: {cap, gloss, text}
        """
        # shuffle all annotations
        random.shuffle(self.annotations)
        count = 0
        data = []

        for obj in self.annotations:
            if count > size:
                break
        
            vid_path = os.path.join(path, obj["name"]) + ".mp4"
            cap = cv2.VideoCapture(vid_path)
            ret = True
            frames = []

            while ret:
                ret, img = cap.read()
                if ret:
                    frames.append(img)

            # Check if the video exists
            if len(frames) == 0:
                continue

            frames = np.array(frames)

            # Apply pre_processing function
            if pre_processing:
                frames = pre_processing(frames)
            
            count += 1
            data.append({'path': vid_path, 'frames': frames, 'gloss': obj["gloss"], "text": obj["text"]})

        return data