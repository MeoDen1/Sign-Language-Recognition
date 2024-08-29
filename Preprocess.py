import numpy as np
import tensorflow as tf

import tf_keras as tfk
from tf_keras import layers

# Clear structure of preprocessing function


USE_TYPES = ['left_hand', 'pose', 'right_hand']

LANDMARKS_PER_FRAME = 543
INPUT_SIZE = 64

START_IDX = 468

LIP_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])

LEFT_HAND_IDXS0 = np.arange(468,489)
RIGHT_HAND_IDXS0 = np.arange(522,543)

LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate((LIP_IDXS0, LEFT_HAND_IDXS0, LEFT_POSE_IDXS0))
LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate((LIP_IDXS0, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0))

N_COLS = len(LANDMARK_IDXS_LEFT_DOMINANT0)
N_DIMS = 2

# Landmark indices after preprocess
LIP_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LIP_IDXS0)).squeeze()
HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_HAND_IDXS0)).squeeze()
POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_POSE_IDXS0)).squeeze()



#########################################################


class PreprocessLayer(layers.Layer):
    """
    Preprocess Layer: only using lips, hands, arm pose coordinates are used
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        normalisation_correction = np.array([
            # X coordinates
            [0] * len(LIP_IDXS) + [1] * len(HAND_IDXS) + [1] * len(POSE_IDXS),
            # Y coordinates
            [0] * len(LANDMARK_IDXS_LEFT_DOMINANT0)
        ], dtype=np.float32)
        # Normalisation_correction shape: (2, 66)
        self.normalisation_correction = tf.transpose(normalisation_correction, [1, 0])
        # Normalisation_correction shape: (66, 2)
        self.normalisation_correction = tf.expand_dims(self.normalisation_correction, axis=0)
        # Output shape: (1, 66, 2)
        
    
    def call(self, x):
        # Input shape: (frame_len, 543, 2)
        # Find dominant hand by comparing hand has less NaN value
        nan_lhand = tf.reduce_sum(tf.cast(tf.math.is_nan(tf.gather(x, LEFT_HAND_IDXS0, axis=1)), dtype=tf.int32))
        nan_rhand = tf.reduce_sum(tf.cast(tf.math.is_nan(tf.gather(x, RIGHT_HAND_IDXS0, axis=1)), dtype=tf.int32))
        
        DOMINANT_HAND_IDXS0 = LEFT_HAND_IDXS0 if nan_lhand <= nan_rhand else RIGHT_HAND_IDXS0
        
        # Get frame which dominant hand is non NaN
        frame_with_dominant_hand = tf.gather(x, DOMINANT_HAND_IDXS0, axis=1)
        # Output shape: (frame_len, len(DOMINANT_HAND_IDXS0), 2)
        
        # Get each nan value, replace it with 0, else replace with 1
        frame_with_dominant_hand = tf.where(tf.math.is_nan(frame_with_dominant_hand), 0, 1)
        # Output shape: (frame_len, len(DOMINANT_HAND_IDXS0), 2)
        
        frame_with_dominant_hand = tf.reduce_sum(frame_with_dominant_hand, axis=[1, 2])
        # Output shape: (frame_len,)
        
        # Get the indices of frame with dominant hand in x
        frame_with_dominant_hand_idxs = tf.where(frame_with_dominant_hand > 0)
        # Output shape: (new_frame_len, 1)
        
        frame_with_dominant_hand_idxs = tf.squeeze(frame_with_dominant_hand_idxs, axis=-1)
        # Output shape: (new_frame_len, )
        
        x1 = tf.gather(x, frame_with_dominant_hand_idxs, axis=0)
        # Output shape: (new_frame_len, 543, 2)
        
        # Gather dominant landmarks
        if nan_lhand <= nan_rhand:
            x1 = tf.gather(x1, LANDMARK_IDXS_LEFT_DOMINANT0, axis=1)
        else:
            x1 = tf.gather(x1, LANDMARK_IDXS_RIGHT_DOMINANT0, axis=1)
            # Convert right_hand into left_hand my multiply with -1 in x-axis and plus 1
            x1 = x1 * tf.where(self.normalisation_correction != 0, -1.0, 1.0) + self.normalisation_correction
        
        # Output shape: (new_frame_len, len(LANDMARK_IDXS_LEFT_DOMINANT)=66, 2)
        
        # Fill NaN value with 0
        x1 = tf.where(tf.math.is_nan(x1), 0.0, x1)
        # Padding video
        if x1.shape[0] < INPUT_SIZE:
            x1 = tf.pad(x1, [[0, INPUT_SIZE - x1.shape[0]], [0, 0], [0, 0]], constant_values=0)
        else:
            # Temp solution
            x1 = x1[:INPUT_SIZE, :, :]
            
        return x1, frame_with_dominant_hand_idxs
    
