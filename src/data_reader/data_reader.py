import wfdb
import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os


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

    def get_csv(self):
        return pd.read_csv(self.database_path + 'ptbxl_database.csv', index_col='ecg_id')

    def get_csv_row(self, row):
        num_skipped_rows = 1 + (row - 1)
        return pd.read_csv(self.database_path + 'ptbxl_database.csv', skiprows=num_skipped_rows, nrows=1, header=None)


    def get_record(self, row: int, **kwargs):
        ''' 
        kwargs Parameters: 
            - freq: can be low (fs = 100 Hz) or high (fs = 500 Hz)
                - options: ['low', 'high']
                - default is high
        '''
        Y = self.get_csv_row(row)

        HR_FILENAME_COLUMN = 27
        LR_FILE_NAME_COLUMN = 26
        freq = kwargs.get('freq', 'high')


        print(Y)
        filename = ''
        if freq == 'high':
            filename = Y.loc[0, HR_FILENAME_COLUMN]
        elif freq == 'low':
            filename = Y.loc[0, LR_FILE_NAME_COLUMN]
        else:
            raise ValueError("freq can take value 'low' or 'high'")


        record = wfdb.rdrecord(self.database_path + filename)
        return record

    def num_records(self) -> int:
        Y = pd.read_csv(self.database_path + 'ptbxl_database.csv', index_col='ecg_id')
        return Y.shape[0]

    def get_record_values(self, record: wfdb.Record | wfdb.MultiRecord):
        return record.p_signal

if __name__ == "__main__":
    reader = PTB_XL_Reader()
    record = reader.get_record(4)       # shape is 5000, 12
                                        # 500 Hz * 10 s = 5000 recordings
                                        # 12 leads

