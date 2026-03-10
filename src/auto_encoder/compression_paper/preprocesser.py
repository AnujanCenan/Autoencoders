import numpy as np
import os
import wfdb
from scipy import signal
from scipy.interpolate import CubicSpline

class ECGDataPreprocessor:
    def __init__(self, target_fs=500, frame_size=1024, hop_size=512):
        self.target_fs = target_fs
        self.frame_size = frame_size
        self.hop_size = hop_size

    def apply_filters(self, data, fs):
        """Removes baseline wander (High-pass) and Power Line Interference (Notch)."""
        # 1. 4th-Order Butterworth High-pass (Cutoff 0.5Hz for Baseline Wander)
        sos_high = signal.butter(4, 0.5, 'hp', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos_high, data)

        # 2. Notch Filter (50Hz for PLI)
        b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs)
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        
        return filtered

    def process_record(self, record_path):
        """Pipeline: Load -> Filter -> Normalize -> Resample -> Frame."""
        # Load Raw V6 Lead
        signals, meta = wfdb.rdsamp(record_path)
        v6_lead = signals[:, 11]
        fs = meta['fs']
        print(f"\tprocessing file {record_path}")
        # --- STEP 1: FILTERING ---
        print(f"\t\t applying filters...")
        clean_v6 = self.apply_filters(v6_lead, fs)
        

        # --- STEP 2: MIN-MAX NORMALIZATION ---
        # Squashing values to [0, 1] for the Sigmoid activation function
        v_min = np.min(clean_v6)
        v_max = np.max(clean_v6)
        # Apply normalization formula: (x - min) / (max - min)
        norm_v6 = (clean_v6 - v_min) / (v_max - v_min + 1e-8)

        # --- STEP 3: RESAMPLING (Cubic Spline) ---
        duration = len(norm_v6) / fs
        time_old = np.linspace(0, duration, len(norm_v6))
        time_new = np.linspace(0, duration, int(duration * self.target_fs))
        
        cs = CubicSpline(time_old, norm_v6)
        resampled_v6 = cs(time_new)

        # --- STEP 4: FRAMING ---
        frames = []
        for i in range(0, len(resampled_v6) - self.frame_size, self.hop_size):
            frames.append(resampled_v6[i : i + self.frame_size])
            
        return np.array(frames)
