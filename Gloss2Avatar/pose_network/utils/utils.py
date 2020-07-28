import os, sys
import numpy as np
import random
import soundfile as sf
import librosa
import ffmpeg
import pdb
import time
import struct


import torch
from torch.utils.data import Dataset

class PoseInterpolatorFCDataset(Dataset):
    """Pose Prediction Dataset."""
    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list of strings): List of paths to the pose npy files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Outputs:
            sample (dict): Sample of our dataset will be a dict
        """        
        self.file_paths = file_paths
        self.transform = transform
    
    def __getitem__(self, index):
        data = np.load(self.file_paths[index])
        data = np.transpose(data, (2, 0, 1))

        sample = {'data': data, 
            'file_name': self.file_paths[index], 
            }
        
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.file_paths) # Should equal number of clean paths...        

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Args:
            sample (dict): A dictionary containing our data. 

        Returns:
            sample (dict): Dictionary data converted to torch tensors
        """
        for dict_key in sample.keys():
            if type(sample[dict_key]) != str: # don't want to run tensorizationon the file name
                sample[dict_key] = torch.from_numpy(sample[dict_key].astype('float32'))
        return sample

class PoseSubsampler(object):
    """Subsamples a section of data to generate input_data and target_output"""

    def __init__(self, edge_sample_length, sample_gap):
        self.edge_sample_length = edge_sample_length
        self.sample_gap = sample_gap

    def __call__(self, sample):
        """
        Args:
            sample (dict): A dictionary containing our data. 
        Returns:
            sample (dict): A dictionary with input_data and target_output sampled from data
        """
        # start_time = time.perf_counter()
        total_seq_frames = sample['data'].shape[0]
        num_input_frames = 2*self.edge_sample_length
        num_output_frames = self.sample_gap
        last_possible_frame = total_seq_frames-num_output_frames-num_input_frames # inclusive of this
        if last_possible_frame < 1:
            pdb.set_trace()
        start_frame = random.randint(0, last_possible_frame)

        relevant_data = sample['data'][start_frame:start_frame+num_output_frames+num_input_frames, :, :]
        target_output_single = relevant_data[self.edge_sample_length:-self.edge_sample_length, :, : ]        
        input_data_single = np.concatenate((relevant_data[:self.edge_sample_length, :, :], relevant_data[-self.edge_sample_length:, :, :]), axis=0)

        sample = {'input_data': input_data_single, 'target_output': target_output_single}
        
        return sample
