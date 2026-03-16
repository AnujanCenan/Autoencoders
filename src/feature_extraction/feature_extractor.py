import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Feature_Extractor(Model):
    def __init__(self):
        super(Feature_Extractor, self).__init__()

        layers.Input(shape=(5000, 1)) # Change 12 to 1

        # --- Initial Encoder Layers (The "Feature Extractor") ---
        self.enc1 = layers.Conv1D(64, kernel_size=7, padding='same', strides=2, activation='relu')
        self.enc2 = layers.Conv1D(32, kernel_size=7, padding='same', strides=2, activation='relu')
        self.enc3 = layers.Conv1D(32, kernel_size=7, padding='same', strides=2, activation='relu')
        self.enc4 = layers.Conv1D(32, kernel_size=7, padding='same', strides=2, activation='relu')
        self.enc5 = layers.Conv1D(16, kernel_size=7, padding='same', strides=2, activation='relu')

        # --- Parallel Bottleneck Layers ---
        # Each branch has 16 filters and kernel size 7 
        self.bn1 = layers.Conv1D(16, kernel_size=7, padding='same', activation='relu', name='bn_1')
        self.bn2 = layers.Conv1D(16, kernel_size=7, padding='same', activation='relu', name='bn_2')
        self.bn3 = layers.Conv1D(16, kernel_size=7, padding='same', activation='relu', name='bn_3')
        self.bn4 = layers.Conv1D(16, kernel_size=7, padding='same', activation='relu', name='bn_4')

        self.concat = layers.Concatenate(axis=-1)
        
        # --- Decoder Layers (Upsampling) ---
        self.dec1 = layers.Conv1DTranspose(16, kernel_size=7, strides=2, padding='same', activation='relu')
        self.dec2 = layers.Conv1DTranspose(32, kernel_size=7, strides=2, padding='same', activation='relu')
        self.dec3 = layers.Conv1DTranspose(32, kernel_size=7, strides=2, padding='same', activation='relu')
        self.dec4 = layers.Conv1DTranspose(64, kernel_size=7, strides=2, padding='same', activation='relu')
        self.dec5 = layers.Conv1DTranspose(1, kernel_size=7, strides=2, padding='same', activation='sigmoid')

    def call(self, x):
        
        # Encoder flow
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        e5 = self.enc5(x)

        # Bottleneck branches
        c1 = self.bn1(e5)
        c2 = self.bn2(e5)
        c3 = self.bn3(e5)
        c4 = self.bn4(e5)

        # Combine for decoder
        latent = self.concat([c1, c2, c3, c4])

        # Decoder flow
        z = self.dec1(latent)
        z = self.dec2(z)
        z = self.dec3(z)
        z = self.dec4(z)
        reconstruction = self.dec5(z)

        return reconstruction, c1, c2, c3, c4
    
    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction, c1, c2, c3, c4 = self(x, training=True)
            
            # 1. Reconstruction Loss (MSE)
            mse_loss = tf.reduce_mean(tf.square(x - reconstruction))
            
            # 2. Sparsity Penalties (L1 Norms from Equation 3)
            # Alphas: a1=0, a2=1e-6, a3=1e-5, a4=1e-4
            # l1_penalty = (
            #     0.0 * tf.reduce_mean(tf.abs(c1)) +
            #     1e-6 * tf.reduce_mean(tf.abs(c2)) +
            #     1e-5 * tf.reduce_mean(tf.abs(c3)) +
            #     1e-4 * tf.reduce_mean(tf.abs(c4))
            # )

            l1_penalty = 0
            
            total_loss = mse_loss + l1_penalty

        # Compute gradients and apply optimizer
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": total_loss, "mse": mse_loss, "l1": l1_penalty}
    
