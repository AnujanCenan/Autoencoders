import tensorflow as tf
from tensorflow.keras import layers, Model

import numpy as np
import os
from dotenv import load_dotenv

from src.feature_extraction.feature_extractor import Feature_Extractor


load_dotenv()

X_train = np.load(os.getenv("V6_TRAINING_DATA"))

print(X_train.shape)        # Output: (69878, 1024)
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, axis=-1)

model = Feature_Extractor()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001, 
    beta_1=0.9, 
    beta_2=0.999
)

model.compile(optimizer=optimizer)
        
model.fit(X_train, epochs=2000, batch_size=32)

model.save_weights("v6_compression_paper.weights.h5")

