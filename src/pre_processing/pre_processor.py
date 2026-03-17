import numpy as np
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler

from scipy.interpolate import CubicSpline

from data_reader import PTB_XL_Reader
from plotter import Plotter


class Pre_Processor:
    def __init__(self):
        pass

    def apply_butterworth(self, data, sampling_freq, cutoff_freq=0.5):

        # data = record.p_signal

        # fs = record.fs          # Sampling frequency

        nyq_freq = 0.5 * sampling_freq
        normalised_cutoff = cutoff_freq / nyq_freq

        ORDER = 4
        sos = signal.butter(ORDER, normalised_cutoff, btype='high', analog=False, output='sos')

        def apply_sos_on_col(col):
            return signal.sosfiltfilt(sos, col)
        
        final_filtered = np.apply_along_axis(apply_sos_on_col, axis=0, arr=data)

        return final_filtered


    def apply_notch_filter(self, data, sampling_freq, notch_freq=50, Q=30) :
        '''
        Default notch_freq and Q are taken from the aBcDe paper
        '''
        
        b_notch, a_notch = signal.iirnotch(notch_freq, Q, sampling_freq)

        def apply_filt_filt_on_col(data_col):
            return signal.filtfilt(b_notch, a_notch, data_col)
        

        final_filtered = np.apply_along_axis(apply_filt_filt_on_col, axis=0, arr=data)

        return final_filtered


    def min_max_normalise(self, data):

        scaler = MinMaxScaler(feature_range=(0, 1))
        # ecg_normalized = scaler.fit_transform(ecg_data)


        def use_scaler(data_col):
            data_col = data_col.reshape(-1, 1)      # fit_transform expects a 2D array
            return scaler.fit_transform(data_col).flatten() # need to flatten back to 1D array for apply_along_axis
        
        normalised_data = np.apply_along_axis(use_scaler, axis=0, arr=data)

        return normalised_data
    
    def resample(self, data, curr_fs, target_fs=500):

        if target_fs == curr_fs:
            return data

        num_recordings = data.shape[0]
        num_leads = data.shape[1]

        duration = num_recordings / curr_fs
        time_old = np.linspace(0, duration, num_recordings)


        new_samples = int(duration * target_fs)
        time_new = np.linspace(0, duration, new_samples)

        # 3. Apply Cubic Spline Resampling
        resampled_ecg_data = np.zeros((new_samples, num_leads))

        for i in range(num_leads):
            cs = CubicSpline(time_old, data[:, i], bc_type='natural')
            resampled_ecg_data[:, i] = cs(time_new)

        return resampled_ecg_data
    

    def clean(self, data, sampling_freq):
        data = self.apply_butterworth(data, sampling_freq)
        data = self.apply_notch_filter(data, sampling_freq)
        data = self.min_max_normalise(data)
        data = self.resample(data, sampling_freq)

        return data


if __name__ == "__main__":
    data_reader = PTB_XL_Reader()
    print("Reading in all data into RAM...")
    num_records = data_reader.get_num_records()
    X, fs = data_reader.get_all_raw_voltages()


    pre_processor = Pre_Processor()
    plotter = Plotter()

    print("Cleaning training dataset...")
    for i in range(num_records):
        # plotter.plot_raw_voltages_mult_leads(X[i])
        X[i] = pre_processor.clean(X[i], fs)
        # plotter.plot_raw_voltages_mult_leads(X[i])

    np.save('cleaned_PTBXL_data.npy', X)

    

    
