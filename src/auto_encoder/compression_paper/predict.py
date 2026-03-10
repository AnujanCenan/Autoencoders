import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from dotenv import load_dotenv
from src.auto_encoder.compression_paper.compression import SparseECGAutoencoder
from src.auto_encoder.compression_paper.preprocesser import ECGDataPreprocessor


# 1. Re-initialize the architecture
# Ensure alpha matches what you used during training
model = SparseECGAutoencoder()
preprocessor = ECGDataPreprocessor(target_fs=500, frame_size=1024, hop_size=512)


# 2. Build and Load
# We provide the input shape so the layers "initialize" before loading weights
model.build((None, 1024, 1)) 
model.load_weights('/Users/unswaccount/Documents/Studying/Year 4/Term 1/Thesis A/Mini-Projects/Autoencoders/src/auto_encoder/compression_paper/model/ecg_sparse_ae.weights.h5') 

print("Model loaded successfully!")

# extract data from directory
load_dotenv()
base_path = os.getenv("PTB_XL_DIRECTORY")
metadata_path = os.path.join(base_path, 'ptbxl_database.csv')

df = pd.read_csv(metadata_path, index_col='ecg_id')


# Assuming you still have your 'preprocessor' object and 'df' metadata
test_records = df[df.strat_fold == 10].iloc[:3] # Let's take the first 3 records

all_test_frames = []

for _, row in test_records.iterrows():
    path = os.path.join(base_path, row.filename_hr)
    # The preprocessor handles the Butterworth, Notch, and Normalization
    frames = preprocessor.process_record(path)
    all_test_frames.append(frames)

# Combine them into one array
X_test_samples = np.vstack(all_test_frames)
X_test_samples = np.expand_dims(X_test_samples, axis=-1)

print(f"Prepared {len(X_test_samples)} frames for testing.")


# Predict the reconstructions
reconstructed_frames = model.predict(X_test_samples)

# Plotting the first 3 frames
plt.figure(figsize=(15, 10))

for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(X_test_samples[i, :, 0], color='royalblue', label='Original V6 (Cleaned)', alpha=0.8)
    plt.plot(reconstructed_frames[i, :, 0], color='crimson', label='AE Reconstruction', linestyle='--', linewidth=2)
    plt.title(f"ECG Frame {i+1} Comparison")
    plt.xlabel("Samples")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()