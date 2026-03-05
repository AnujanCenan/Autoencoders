import wfdb
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from ..data_reader.data_reader import PTB_XL_Reader
from ..plotter.plotter import Plotter_WFDB


class Pre_Processor:
    def __init__(self):
        pass

    def apply_butterworth(self, record: wfdb.Record, cutoff_freq=0.5):

        data = record.p_signal[:,0]  # 1st lead only
        fs = record.fs          # Sampling frequency

        nyq_freq = 0.5 * fs
        normalised_cutoff = cutoff_freq / nyq_freq


        sos = signal.butter(4, normalised_cutoff, btype='high', analog=False, output='sos')


        filtered_signal = signal.sosfiltfilt(sos, data)
        return filtered_signal    
        
    def apply_band_stop_filter(self):
        pass

    def min_max_normalise(self):
        pass

if __name__ == "__main__":
    data_reader = PTB_XL_Reader()
    record = data_reader.get_record(record_id=1)
    
    plotter = Plotter_WFDB()
    # plotter.plot_sample(record)
    plotter.plot_raw_voltages(record.p_signal[:,0])
    
    pre_processor = Pre_Processor()
    filtered_sig = pre_processor.apply_butterworth(record=record)

    plotter.plot_raw_voltages(filtered_sig, colour="red")




    