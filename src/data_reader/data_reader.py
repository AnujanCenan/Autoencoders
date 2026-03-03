import wfdb
import pandas as pd
import numpy as np



from enum import Enum



class Data_Src(Enum):
    PTB_XL = 0

# Each strategy should have a get_record method which returns a record
class PTB_XL_Reader():
    def __init__(self, database_path):
        self.database_path = database_path


    def get_record(self, record_id: int):
        Y = pd.read_csv(self.database_path + 'ptbxl_database.csv', index_col='ecg_id')

        print(Y.loc[1, "filename_lr"])
        filename = Y.loc[record_id, "filename_lr"]
        print(self.database_path + filename)

        record = wfdb.rdrecord(self.database_path + filename)
        return record


class Data_Reader():
    def __init__(self, strategy: Data_Src, database_path: str):
        
        if (strategy == Data_Src.PTB_XL):
            self.strategy = PTB_XL_Reader(database_path)
        else:
            print("Invalid strategy given to Data_Reader class")
    
    
    def get_record(self, record_id: int):
        return self.strategy.get_record(record_id)


