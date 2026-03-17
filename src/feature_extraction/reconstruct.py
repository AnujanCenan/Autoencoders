import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import os
from dotenv import load_dotenv

from src.feature_extraction.feature_extractor import Feature_Extractor

load_dotenv()
clean_data_dir = os.getenv("V6_TESTING_DATA")
X = np.load(clean_data_dir)

# 1. Pick a random ECG from the test set (which the model never saw during training)
test_index = 42 
input_ecg = X[42:43] # Shape: (1, 5000, 12)

print(f"Data Max: {np.max(input_ecg[:,0])}")
print(f"Data Min: {np.min(input_ecg[:,0])}")
print(f"Data Mean: {np.mean(input_ecg[:,0])}")

model = Feature_Extractor(latent_dim=128)
model.build(input_shape=(None, 5000, 12))
model.load_weights('/Users/unswaccount/Documents/Studying/Year 4/Term 1/Thesis A/Mini-Projects/Autoencoders/ecg_autoencoder_weights.weights.h5')

# 2. Run the reconstruction
reconstructed_ecg = model.predict(input_ecg)

# 3. Create the plot
# We'll plot Lead I (index 0) and Lead V1 (index 6) to compare
leads_to_plot = [0, 6]
lead_names = ['Lead I', 'Lead V1']

plt.figure(figsize=(15, 10))


for i, lead_idx in enumerate(leads_to_plot):
    plt.subplot(2, 1, i+1)
    
    # Plot Original
    plt.plot(input_ecg[0, :, lead_idx], label='Original', color='blue', alpha=0.6)
    
    # Plot Reconstructed
    plt.plot(reconstructed_ecg[0, :, lead_idx], label='Reconstructed', 
             color='red', linestyle='--', linewidth=1.5)
    
    plt.title(f'Comparison: {lead_names[i]}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()