#!/bin/python3

import wfdb
from dotenv import load_dotenv
import os

from ..data_reader.data_reader import Data_Src, Data_Reader

class Plotter_WFDB:

    def initialise_data_reader(self, strategy: Data_Src, database_path: str):
        self.data_reader = Data_Reader(strategy, database_path)

    def plot_sample(self, record_id):
        record = self.data_reader.get_record(record_id)


        wfdb.plot_wfdb(record=record, title=f"Record {record_id} Plot", figsize=(15, 8))
        return

    def plot_all_samples(self):
        num_records = self.data_reader.num_records()

        for id in range(1, num_records + 1):
            self.plot_sample(id)

    def plot_directory(self, dir_path):
        pass


if __name__ == "__main__":
    load_dotenv()

    plotter = Plotter_WFDB()
    plotter.initialise_data_reader(Data_Src.PTB_XL, os.getenv("PTB_XL_DIRECTORY"))

    plotter.plot_all_samples()