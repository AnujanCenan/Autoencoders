# Initialize the model
# alpha controls the trade-off: higher = more compression/sparsity
import tensorflow as tf

from src.auto_encoder.compression_paper.compression import SparseECGAutoencoder
from src.auto_encoder.compression_paper.preprocesser import ECGDataPreprocessor

import time

def train_sparse_autoencoder(model, train_data, val_data, epochs=50, batch_size=32):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch+1}")
        start_time = time.time()

        # Training Phase
        for step, x_batch in enumerate(train_dataset):
            print(f"\tBeginning batch {step}...")
            with tf.GradientTape() as tape:
                # 1. Forward pass
                reconstruction = model(x_batch, training=True)
                
                # 2. Compute Primary Loss (MSE)
                main_loss = mse_loss_fn(x_batch, reconstruction)
                
                # 3. Add the L1 internal losses (added via model.add_loss in the class)
                total_loss = main_loss + sum(model.losses)

            # 4. Backpropagation
            grads = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Validation Phase
        val_mse = 0
        for x_val_batch in val_dataset:
            val_reconstruction = model(x_val_batch, training=False)
            val_mse += mse_loss_fn(x_val_batch, val_reconstruction)
        
        val_mse /= len(val_dataset)
        
        print(f"Epoch {epoch+1} done. Loss: {total_loss:.6f}, Val MSE: {val_mse:.6f}")
        print(f"Time taken: {time.time() - start_time:.2f}s")

    return model

import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

# Paths (Update these to your local directories)
load_dotenv()
base_path = os.getenv("PTB_XL_DIRECTORY")
metadata_path = os.path.join(base_path, 'ptbxl_database.csv')

# 1. Load Metadata
df = pd.read_csv(metadata_path, index_col='ecg_id')

# 2. Initialize Preprocessor
# target_fs=500 matches PTB-XL high-res and your 1024-sample window logic
preprocessor = ECGDataPreprocessor(target_fs=500, frame_size=1024, hop_size=512)


def get_data_split(df, folds):
    """Helper to process all records in specific folds."""
    subset_df = df[df.strat_fold.isin(folds)]
    
    all_frames = []
    for _, row in subset_df.iterrows():
        file_path = os.path.join(base_path, row.filename_hr) # filename_hr is the 500Hz path
        
        # This function (from our previous code) handles: 
        # Butterworth -> Notch -> MinMax -> Cubic Spline -> Framing
        frames = preprocessor.process_record(file_path)
        all_frames.append(frames)
    
    # Stack everything into (Total_Frames, 1024, 1)
    X = np.vstack(all_frames)
    return np.expand_dims(X, axis=-1)

# Apply the split
print("Processing Training Data...")
X_train = get_data_split(df, range(1, 9))  # Folds 1-8

print("Processing Validation Data...")
X_val = get_data_split(df, [9])            # Fold 9

# # To use it later in a different script:
# new_model = SparseECGAutoencoder(alpha=0.001)
# # We need to call the model on a dummy input first to "build" the layers
# new_model.build((None, 1024, 1)) 
# new_model.load_weights('ecg_sparse_ae_weights.h5')


# 1. Create the Model

print(f"Constructing base architecture...")
model = SparseECGAutoencoder()

# 2. Run the training loop function we wrote
# We pass X_train and X_val into the function
print(f"Beginning training...")
trained_model = train_sparse_autoencoder(
    model=model, 
    train_data=X_train, 
    val_data=X_val, 
    epochs=50, 
    batch_size=32  # 32 frames per gradient update
)

trained_model.save_weights('ecg_sparse_ae.weights.h5')


import matplotlib.pyplot as plt

# Take one frame from the validation set
original_sample = X_val[0:1] 
reconstructed_sample = trained_model.predict(original_sample)

plt.figure(figsize=(12, 4))
plt.plot(original_sample[0, :, 0], label='Original V6 (Cleaned)')
plt.plot(reconstructed_sample[0, :, 0], label='Reconstructed', linestyle='--')
plt.title("V6 Lead: Original vs. Sparse Reconstruction")
plt.legend()
plt.show()
