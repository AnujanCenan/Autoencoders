import numpy as np
from dotenv import load_dotenv
import os

class Framer:
    '''
    given ECG data that has more than n recordings, returns a list of ECG data
    'frames', each with with n recordings in it by passing a window of size n over 
    the original data

    Currently allows for only one lead worth of data
    parameters in the method can be adjusted to determine if the new ECG data should
    be overlapping or not (and 'how' overlapping they should be)

    Note that the frames are returned in a numpy array
    '''
    def get_frames_1_lead(self, data, window_size=1024, jump=1024):
        # data should have a size of (N, ) 
        print(data.shape)  
        total_num_samples = data.shape[0]       # should be 5000 during the debugging phase
        total_num_frames = (total_num_samples - window_size) // jump + 1      # num_frames = number of possible jumps + 1
        print(f"total_num_frames={total_num_frames}")
        
        left = 0
        right = window_size

        if right >= total_num_samples:
            raise ValueError(f"Length of data={total_num_samples} needs to be bigger than window size={window_size}")
        
        all_frames = np.zeros((total_num_frames, window_size))
        curr_frame = 0
        while right < total_num_samples:
            all_frames[curr_frame] = data[left:right]

            left += jump
            right = left + window_size
            curr_frame += 1

        return all_frames, total_num_frames

if __name__ == "__main__":
    load_dotenv()

    clean_data_path = os.getenv("CLEAN_PTB_XL")
    CLEAN_DATA = np.load(clean_data_path)

    print(f"CLEAN DATA shape = {CLEAN_DATA.shape}")     # output: (21837, 5000, 12)
    v6_data = CLEAN_DATA[:, :, 11]
    print(f"v6 data shape = {v6_data.shape}")           # output: (21837, 5000)


    print(v6_data[0])

    framer = Framer()

    FRAME_SIZE = 1024

    total_num_frames =  (v6_data.shape[1] // FRAME_SIZE) * v6_data.shape[0]         # (5000 // 1024) * 21837
    print(total_num_frames)                                                         # 87348
    all_framed_data = np.zeros((total_num_frames, FRAME_SIZE))

    curr_frame_id = 0

    for i in range(v6_data.shape[0]):
        curr_frames = None
        try:
            curr_frames, num_frames = framer.get_frames_1_lead(v6_data[i, :])
        except ValueError:
            exit(1)

        for j in range(num_frames):
            all_framed_data[curr_frame_id, :] = curr_frames[j, :]

            curr_frame_id += 1
    

    print(all_framed_data.shape)        # (87348, 1024)

    # np.save('1024_frame_v6.npy', all_framed_data)     # uncomment if you want to save the file
