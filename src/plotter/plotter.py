#!/bin/python3

import wfdb


import numpy as np

from ..data_reader.data_reader import PTB_XL_Reader

class Plotter_WFDB:
    
    def plot_sample(self, record: wfdb.Record):

        wfdb.plot_wfdb(record=record, title=f"Record Plot", figsize=(15, 8))
        return

if __name__ == "__main__":
    reader = PTB_XL_Reader()
    plotter = Plotter_WFDB()

    plotter.plot_sample(reader.get_record(record_id=1))
