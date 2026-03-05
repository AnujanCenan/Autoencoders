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
        print(data.shape)
        number_of_leads = data.shape[1]

        fs = record.fs          # Sampling frequency

        nyq_freq = 0.5 * fs
        normalised_cutoff = cutoff_freq / nyq_freq


        sos = signal.butter(4, normalised_cutoff, btype='high', analog=False, output='sos')

        final_filtered = np.zeros_like(data)
        for curr_lead in range(number_of_leads):
            data_col = data[:,curr_lead]
            filtered_signal = signal.sosfiltfilt(sos, data_col)
            final_filtered[:,curr_lead] = filtered_signal

        return wfdb.Record(
            p_signal=final_filtered,
            n_sig=number_of_leads,
            fs=record.fs,
            units=record.units,
            sig_name=record.sig_name
        )

    def apply_band_stop_filter(self):
        pass

    def min_max_normalise(self):
        pass

if __name__ == "__main__":
    data_reader = PTB_XL_Reader()
    record = data_reader.get_record(record_id=1)
    
    plotter = Plotter()
    plotter.plot_raw_voltages(record.p_signal[:,0])
    plotter.plot_sample(record)
    
    pre_processor = Pre_Processor()

    filtered_signal_record = pre_processor.apply_butterworth(record=record)


    plotter.plot_raw_voltages_mult_leads(filtered_signal_record.p_signal, "red")
    plotter.plot_sample(filtered_signal_record)
    # for curr_lead in range(12):
    #     plotter.plot_raw_voltages(filtered_signal[:,curr_lead], colour="red")




    