import pickle
import numpy as np
import gzip
import random
import os
import cv2
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

class Dataset:
    def __init__(self) -> None:
        with gzip.open('Dataset/phoenix14t.pami0.train.annotations_only.gzip', 'rb') as f:
            # annotations: list of object
            # - name: path (train/...)
            # - signer: signer name
            # - gloss: JETZT ...
            # - text: ...
            self.annotations = pickle.load(f)

    def load(self, path="Dataset/videos_phoenix/videos", size: int=10) -> list:
        """
        Return format: List of object: {cap, gloss, text}
        """
        # shuffle all annotations
        random.shuffle(self.annotations)
        count = 0
        data = []

        for obj in self.annotations:
            if count >= size:
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

            count += 1
            data.append({'path': vid_path, 'frames': frames, 'gloss': obj["gloss"], "text": obj["text"]})

        return data
    
def frame_processing(frames: np.array, seq_len: int):
    """
    Gray scale the frames and padding the frames to `seq_len`
    `input`: shape (len, height, width, channels)
    """
    output = []
    for i, frame in enumerate(frames):
        if (i == seq_len): 
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output.append(np.expand_dims(frame, axis=-1))

    output += [np.zeros_like(output[0]) for _ in range(seq_len - len(output))]

    return np.array(output) / 255


def pre_processing_1(data, encoder_seq_len: int, decoder_seq_len: int):
    """
    Pre-processing the data

    Parameters
    --
    data: dict
        {'path', 'frames', 'gloss', 'text'}
    encoder_seq_len: int
        Max encoder sequence length
    decoder_seq_len: int
        Max decoder sequence length

    Returns
    --
    tokenizer: Tokenizer
        tokenizer to convert texts into sequence
    frames_data: ndarray
        sequence of videos with shape (size, encoder_seq_len, height, width, 1)
    gloss_data: ndarray
        sequence of token with shape (size, decoder_seq_len)
    text_data: ndarray
        empty

    """
    frames_data = []
    gloss_data = []
    text_data = []

    for obj in data:
        # Pre-processing frame
        frames_data.append(frame_processing(obj['frames'], encoder_seq_len))

        gloss_data.append('SOS ' + obj['gloss'] + ' EOS')
        text_data.append('SOS ' + obj['text'] + ' EOS')

    # Tokenize gloss
    tokenizer = Tokenizer()
    # Update vocabulary
    tokenizer.fit_on_texts(gloss_data)
    # Convert gloss data into sequence of int
    gloss_data = tokenizer.texts_to_sequences(gloss_data)
    # Padding sequence
    gloss_data = pad_sequences(gloss_data, maxlen=decoder_seq_len, padding='post')

    tokenizer.word_index['PAD'] = 0
    tokenizer.index_word[0] = 'PAD'

    # Tokenize text
    # ...

    return tokenizer, np.array(frames_data), np.array(gloss_data), np.array(text_data)