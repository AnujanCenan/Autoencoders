import tensorflow as tf
from tensorflow.keras import layers, Model

class SparseECGAutoencoder(Model):
    def __init__(self):
        super(SparseECGAutoencoder, self).__init__()
        self.alpha = [0, 0.000001, 0.00001, 0.0001] # Regularization coefficient
        
        # --- ENCODER ---
        # Input: 1024-dimensional vector (reshaped to 1024, 1 for Conv1D)
        self.enc_conv1 = layers.Conv1D(64, 7, strides=2, padding='same', activation='relu')
        self.enc_conv2 = layers.Conv1D(32, 7, strides=2, padding='same', activation='relu')
        self.enc_conv3 = layers.Conv1D(32, 7, strides=2, padding='same', activation='relu')
        self.enc_conv4 = layers.Conv1D(32, 7, strides=2, padding='same', activation='relu')
        self.enc_conv5 = layers.Conv1D(16, 7, strides=2, padding='same', activation='relu')
        
        # --- PARALLEL BOTTLENECKS ---
        # Kernel size 7, filter number 16 as specified
        self.bn1 = layers.Conv1D(16, 7, padding='same', activation='relu')
        self.bn2 = layers.Conv1D(16, 7, padding='same', activation='relu')
        self.bn3 = layers.Conv1D(16, 7, padding='same', activation='relu')
        self.bn4 = layers.Conv1D(16, 7, padding='same', activation='relu')
        
        # --- DECODER ---
        # Mirrored Architecture
        self.dec_convT1 = layers.Conv1DTranspose(16, 7, strides=2, padding='same', activation='relu')
        self.dec_convT2 = layers.Conv1DTranspose(32, 7, strides=2, padding='same', activation='relu')
        self.dec_convT3 = layers.Conv1DTranspose(32, 7, strides=2, padding='same', activation='relu')
        self.dec_convT4 = layers.Conv1DTranspose(64, 7, strides=2, padding='same', activation='relu')
        # Final layer: 1 filter (ECG signal), Sigmoid activation
        self.dec_output = layers.Conv1DTranspose(1, 7, strides=2, padding='same', activation='sigmoid')

    def call(self, x):
        # Initial Feature Extraction
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)
        x = self.enc_conv5(x)
        
        # Parallel Bottlenecks (c1, c2, c3, c4)
        c1 = self.bn1(x)
        c2 = self.bn2(x)
        c3 = self.bn3(x)
        c4 = self.bn4(x)
        
        # Apply L1 Regularization manually to the bottlenecks
        self.add_loss(self.alpha[0] * tf.reduce_mean(tf.abs(c1)))
        self.add_loss(self.alpha[1] * tf.reduce_mean(tf.abs(c2)))
        self.add_loss(self.alpha[2] * tf.reduce_mean(tf.abs(c3)))
        self.add_loss(self.alpha[3] * tf.reduce_mean(tf.abs(c4)))
        
        # Combine (Concatenate along the filter/channel axis)
        # Resulting shape: (batch, 32, 64)
        combined = tf.concat([c1, c2, c3, c4], axis=-1)
        
        # Since the first decoder layer expects 16 filters but we have 64 (16*4),
        # we can use a small 1x1 conv to map back or feed directly if adjusted.
        # Based on your paper: "Reshaped into feature vectors"
        # Let's assume the first decoder layer handles the combined input:
        
        y = self.dec_convT1(combined) 
        y = self.dec_convT2(y)
        y = self.dec_convT3(y)
        y = self.dec_convT4(y)
        reconstruction = self.dec_output(y)
        
        return reconstruction