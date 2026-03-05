import wfdb
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

from enum import Enum


    
class Data_Reader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_record(self, record_id: int):
        pass
    
    @abstractmethod
    def num_records(self) -> int:
        pass
    
    @abstractmethod
    def get_record_values(self, record: wfdb.Record | wfdb.MultiRecord):
        pass


class PTB_XL_Reader(Data_Reader):
    def __init__(self):
        load_dotenv()
        self.database_path = os.getenv("PTB_XL_DIRECTORY")


    def get_record(self, record_id: int):
        Y = pd.read_csv(self.database_path + 'ptbxl_database.csv', index_col='ecg_id')

        print(Y.loc[1, "filename_lr"])
        filename = Y.loc[record_id, "filename_lr"]
        print(self.database_path + filename)

        record = wfdb.rdrecord(self.database_path + filename)
        return record

    def num_records(self) -> int:
        Y = pd.read_csv(self.database_path + 'ptbxl_database.csv', index_col='ecg_id')
        print(Y.shape[0])
        return Y.shape[0]

    def get_record_values(self, record: wfdb.Record | wfdb.MultiRecord):
        print(record.p_signal)
        return record.p_signal
