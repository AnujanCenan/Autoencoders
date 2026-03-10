import wfdb
import numpy as np
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler

from scipy.interpolate import CubicSpline


from ..data_reader.data_reader import PTB_XL_Reader
from ..plotter.plotter import Plotter


class Pre_Processor:
    def __init__(self):
        pass

    def apply_butterworth(self, record: wfdb.Record, cutoff_freq=0.5) -> wfdb.Record:

        data = record.p_signal

        fs = record.fs          # Sampling frequency

        nyq_freq = 0.5 * fs
        normalised_cutoff = cutoff_freq / nyq_freq

        ORDER = 4
        sos = signal.butter(ORDER, normalised_cutoff, btype='high', analog=False, output='sos')

        def apply_sos_on_col(col):
            return signal.sosfiltfilt(sos, col)
        
        final_filtered = np.apply_along_axis(apply_sos_on_col, axis=0, arr=data)

        return wfdb.Record(
            p_signal=final_filtered,
            n_sig=record.n_sig,
            fs=record.fs,
            units=record.units,
            sig_name=record.sig_name
        )


    def apply_notch_filter(self, record: wfdb.Record, notch_freq=50, Q=30) -> wfdb.Record:
        '''
        Default notch_freq and Q are taken from the aBcDe paper
        '''
        fs = record.fs
        data = record.p_signal
        
        b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)

        def apply_filt_filt_on_col(data_col):
            return signal.filtfilt(b_notch, a_notch, data_col)
        

        final_filtered = np.apply_along_axis(apply_filt_filt_on_col, axis=0, arr=data)

        return wfdb.Record(
            p_signal=final_filtered,
            n_sig=record.n_sig,
            fs=record.fs,
            units=record.units,
            sig_name=record.sig_name
        )


    def min_max_normalise(self, record: wfdb.Record) -> wfdb.Record:
        
        print(f"\t\tChecking p_sig shape: {record.p_signal.shape}")
        data = record.p_signal

        scaler = MinMaxScaler(feature_range=(0, 1))
        # ecg_normalized = scaler.fit_transform(ecg_data)


        def use_scaler(data_col):
            data_col = data_col.reshape(-1, 1)      # fit_transform expects a 2D array
            return scaler.fit_transform(data_col).flatten() # need to flatten back to 1D array for apply_along_axis
        
        normalised_data = np.apply_along_axis(use_scaler, axis=0, arr=data)

        return wfdb.Record(
            p_signal=normalised_data,
            n_sig=record.n_sig,
            fs=record.fs,
            units=record.units,
            sig_name=record.sig_name
        )
    
    def resample(self, record: wfdb.Record, target_fs=500) -> wfdb.Record:
        fs = record.fs
        if target_fs == fs:
            return record

        num_recordings = record.p_signal.shape[0]
        num_leads = record.p_signal.shape[1]

        duration = num_recordings / fs
        time_old = np.linspace(0, duration, num_recordings)


        new_samples = int(duration * target_fs)
        time_new = np.linspace(0, duration, new_samples)

        # 3. Apply Cubic Spline Resampling
        resampled_ecg_data = np.zeros((new_samples, num_leads))

        for i in range(num_leads):
            cs = CubicSpline(time_old, record.p_signal[:, i], bc_type='natural')
            resampled_ecg_data[:, i] = cs(time_new)

        return wfdb.Record(
            p_signal=resampled_ecg_data,
            n_sig=num_leads,
            fs=target_fs,
            units=record.units,
            sig_name=record.sig_name
        )
    

    def clean(self, record: wfdb.Record) -> wfdb.Record:
        record = self.apply_butterworth(record)
        record = self.apply_notch_filter(record)
        record = self.min_max_normalise(record)

        return record


if __name__ == "__main__":

    print(f"Running Preprocessor Program...")
    data_reader = PTB_XL_Reader()
    record = data_reader.get_record(row=3, freq='low')
    print(record.p_signal.shape)

    
    plotter = Plotter()
    plotter.plot_sample(record)
    
    pre_processor = Pre_Processor()
    
    print("Cleaning record")
    record = pre_processor.clean(record)

    record = pre_processor.resample(record)

    print(record.p_signal.shape)

    plotter.plot_sample(record)
