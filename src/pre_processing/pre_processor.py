import wfdb
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

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

        order = 4
        sos = signal.butter(order, normalised_cutoff, btype='high', analog=False, output='sos')

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


    def apply_notch_filter(self, record: wfdb.Record, notch_freq=50, Q=30):
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


    def min_max_normalise(self):
        pass

if __name__ == "__main__":

    print(f"Running Preprocessor Program...")
    data_reader = PTB_XL_Reader()
    record = data_reader.get_record(record_id=1)
    
    plotter = Plotter()
    plotter.plot_raw_voltages(record.p_signal[:,0])
    plotter.plot_sample(record)
    
    pre_processor = Pre_Processor()
    
    print("\tApplying butterworth filter for baseline wander removal...")
    filtered_signal_record = pre_processor.apply_butterworth(record=record)
    
    print("\tApplying notch filter for power line interference removal (assumes notch_freq=50Hz)...")
    filtered_signal = pre_processor.apply_notch_filter(record=filtered_signal_record)

    plotter.plot_sample(filtered_signal)

    # plotter.plot_raw_voltages_mult_leads(filtered_signal_record.p_signal, "red")
    # plotter.plot_sample(filtered_signal_record)



    