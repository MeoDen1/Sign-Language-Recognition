import numpy as np
import tensorflow as tf
import math

import tf_keras as tfk
from tf_keras import layers, models
from tf_keras.losses import categorical_crossentropy

# Clear structure of model implementation


####################################################################
# CONFIG

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

NUM_CLASSES = 250

# Model config
LAYER_NORM_EPS = 1e-6

# Dense layer units for landmarks
LIPS_UNITS = 384
HANDS_UNITS = 384
POSE_UNITS = 384
# final embedding and transformer embedding size
UNITS = 512

# Transformer
NUM_BLOCKS = 2
MLP_RATIO = 2

# Dropout
EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
CLASSIFIER_DROPOUT_RATIO = 0.10

# Initiailizers
INIT_HE_UNIFORM = tfk.initializers.he_uniform
INIT_GLOROT_UNIFORM = tfk.initializers.glorot_uniform
INIT_ZEROS = tfk.initializers.constant(0.0)

GELU = tfk.activations.gelu


###################################################################################


# LANDMARK EMBEDDING
class LandmarkEmbedding(layers.Layer):
    def __init__(self, units, name):
        super().__init__(name=f'{name}_embedding')
        
        self.dense = models.Sequential([
            layers.Dense(UNITS, name=f'{name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM),
            layers.Activation(GELU),
            layers.Dense(UNITS, name=f'{name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM)
        ], name=f'{name}_dense')
        
    def call(self, x):
        # Input shape: (batch_size, INPUT_SIZE, -1)
        return self.dense(x)
        # Output shape: (batch_size, INPUT_SIZE, UNITS)


# EMBEDDING LAYER
class Embedding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Embedding layer
        self.lip_embedding = LandmarkEmbedding(LIPS_UNITS, name='lips')
        self.hand_embedding = LandmarkEmbedding(HANDS_UNITS, name='hand')
        self.pose_embedding = LandmarkEmbedding(POSE_UNITS, name='pose')
        
        # Landmark Weights
        self.landmark_weights = tf.Variable(tf.zeros([3], dtype=tf.float32), name='landmark_weights')
        
        self.fc = models.Sequential([
            layers.Dense(UNITS, name='fc_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM),
            layers.Activation(GELU),
            layers.Dense(UNITS, name='fc_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM)
        ], name='fc')
        
        self.positional_encoding = tf.expand_dims(self.get_positional_encoding(INPUT_SIZE, UNITS), axis=0)
        # PE shape: (1, INPUT_SIZE, UNITS)
    
    # Using sinusoid positional encoding
    def get_positional_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in range(int(d / 2)):
                denominator = pow(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        
        return tf.convert_to_tensor(P, dtype=tf.float32)
    
    def call(self, lip, hand, pose):
        lip_output = self.lip_embedding(lip)
        # Output shape: (batch_size, INPUT_SIZE, UNITS)
        
        hand_output = self.hand_embedding(hand)
        # Output shape: (batch_size, INPUT_SIZE, UNITS)
        
        pose_output = self.pose_embedding(pose)
        # Output shape: (batch_size, INPUT_SIZE, UNITS)
        
        # Stack to full landmark
        x = tf.stack((lip_output, hand_output, pose_output), axis=-1)
        # Output shape: (batch_size, INPUT_SIZE, UNITS, 3)
        
        x = x * tf.math.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=-1)
        # Output shape: (batch_size, INPUT_SIZE, UNITS)
        
        x = self.fc(x)
        # Output shape: (batch_size, INPUT_SIZE, UNITS)
        
        return x + self.positional_encoding
    

# MULTIHEAD ATTENTION
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_of_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        
        self.depth = d_model // num_of_heads
        
        self.wq = [layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [layers.Dense(self.depth) for i in range(num_of_heads)]
        
        self.wo = layers.Dense(d_model)

    # scaled dot_product
    def scaled_dot_product(self, queries, keys, values, mask=None):
        scores = tf.matmul(queries, keys, transpose_b=True) 
        scores /= tf.math.sqrt(tf.cast(queries.shape[-1], dtype=tf.float32))

        if mask is not None:
            scores += -1e9 * mask

        weights = tf.math.softmax(scores)

        return tf.matmul(weights, values)
        
    def call(self,x, attention_mask):
        multi_attn = []
        
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(self.scaled_dot_product(Q, K, V, attention_mask))
            
        multi_head = tf.concat(multi_attn,axis=-1)
        multi_head_attention = self.wo(multi_head)
        
        return multi_head_attention
    

# TRANSFORMER
class Transformer(models.Model):
    def __init__(self, num_blocks):
        super().__init__(name='transformer')
        self.num_blocks = num_blocks
    
    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(UNITS, 8))
            
            # Multi Layer Perception
            self.mlps.append(models.Sequential([
                layers.Dense(UNITS * MLP_RATIO, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                layers.Dropout(MLP_DROPOUT_RATIO),
                layers.Dense(UNITS, kernel_initializer=INIT_HE_UNIFORM),
            ]))
        
    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x, attention_mask)
            x = x + mlp(x)
    
        return x
    

def scce_with_ls(y_true, y_pred):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)
    

def build_model():
    landmark_input = layers.Input([INPUT_SIZE, N_COLS, N_DIMS], name='landmark_input')
    x = landmark_input
    # x shape: (batch_size, INPUT_SIZE, N_COLS=66, N_DIMS=2)
    
    # Create mask (1 as padding value, 0 for non-padding value)
    mask = tf.equal(x, 0)
    mask = tf.cast(tf.reduce_all(mask, axis=[2, 3]), tf.float32)
    # mask shape: (batch_size, INPUT_SIZE)
    
    temp_mask = tf.expand_dims(mask, axis=1)
    # mask1 shape: (batch_size, 1, INPUT_SIZE)
    
    mask = tf.expand_dims(mask, axis=-1)
    # mask shape: (batch_size, INPUT_SIZE, 1)
    
    attention_mask = tf.where(mask + temp_mask > 0, 1.0, 0.0)
    # attention_mask shape: (batch_size, INPUT_SIZE, INPUT_SIZE)
    
    
    ############################################
    # Get landmark parts
    
    ############################################
    # LIP
    lip_landmark = tf.slice(x, [0, 0, 0, 0], [-1, -1, 40, -1])
    # Output shape: (batch_size, INPUT_SIZE, 40, 2)
    
    lip_landmark_mean = tf.reduce_mean(lip_landmark, axis=2, keepdims=True)
    lip_landmark_std = tf.math.reduce_std(lip_landmark, axis=2, keepdims=True)
    # Lip landmark mean & std shape: (batch_size, INPUT_SIZE, 1, 2)
    
    
    # NOTE: When applying Z-score normalization in each frame, the standard deviation 
    # of padding value is 0, which divided by 0 will lead to NaN. So for every std value
    # we will replace it with 1
    lip_landmark_std = tf.where(lip_landmark_std == 0, 1.0, lip_landmark_std)
    
    lip_landmark = tf.reshape(
        (lip_landmark - lip_landmark_mean) / lip_landmark_std, 
        [-1, INPUT_SIZE, 40*2]
    )
    
    
    # Output shape: (batch_size, INPUT_SIZE, 80)
    
    
    ############################################
    # HAND
    hand_landmark = tf.slice(x, [0, 0, 40, 0], [-1, -1, 21, -1])
    # Output shape: (batch_size, INPUT_SIZE, 21, 2)
    
    hand_landmark_mean = tf.reduce_mean(hand_landmark, axis=2, keepdims=True)
    hand_landmark_std = tf.math.reduce_std(hand_landmark, axis=2, keepdims=True)
    # hand landmark mean & std shape: (batch_size, INPUT_SIZE, 1, 2)
    hand_landmark_std = tf.where(hand_landmark_std==0, 1.0, hand_landmark_std)
    
    hand_landmark = tf.reshape(
        (hand_landmark - hand_landmark_mean) / hand_landmark_std, 
        [-1, INPUT_SIZE, 21*2]
    )
    # Output shape: (batch_size, INPUT_SIZE, 42)
    
    
    ############################################
    # POSE
    pose_landmark = tf.slice(x, [0, 0, 61, 0], [-1, -1, 5, -1])
    # Output shape: (batch_size, INPUT_SIZE, 5, 2)
    
    pose_landmark_mean = tf.reduce_mean(pose_landmark, axis=2, keepdims=True)
    pose_landmark_std = tf.math.reduce_std(pose_landmark, axis=2, keepdims=True)
    # pose landmark mean & std shape: (batch_size, INPUT_SIZE, 1, 2)
    pose_landmark_std = tf.where(pose_landmark_std==0, 1.0, pose_landmark_std)
    
    pose_landmark = tf.reshape(
        (pose_landmark - pose_landmark_mean) / pose_landmark_std, 
        [-1, INPUT_SIZE, 5*2]
    )
    # Output shape: (batch_size, INPUT_SIZE, 10)

    ############################################
    
    x = Embedding()(lip_landmark, hand_landmark, pose_landmark)
    # Output shape: (batch_size, INPUT_SIZE, UNITS)
    
    x = Transformer(NUM_BLOCKS)(x, attention_mask)
    # Output shape: (batch_size, INPUT_SIZE, UNITS)
    
    # Pooling
    # Mark non-padding as 1 and padding value as zero
    mask = tf.where(mask==0, 1.0, 0.0)
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    
    # Classifier Dropout
    x = layers.Dropout(CLASSIFIER_DROPOUT_RATIO)(x)
    
    # Classification Layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_initializer=INIT_GLOROT_UNIFORM)(x)
    
    # Create model
    model = models.Model(inputs=landmark_input, outputs=outputs)
    
    # Sparse Categorical Cross Entropy With Label Smoothing
    loss = scce_with_ls
    
    # Adam Optimizer with weight decay
    optimizer = tfk.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    
    # TopK Metrics
    metrics = [
        tfk.metrics.SparseCategoricalAccuracy(name='acc'),
        tfk.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tfk.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model