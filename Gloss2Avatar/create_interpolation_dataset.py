import torch
from tqdm import tqdm
import os
import numpy as np

num_intermediate_frames = 10
dataset_x = []
dataset_y = []

# Load data
data_dir = 'data/RawData/'
avatar_data_dir = os.path.join(data_dir, 'npy_dir')
pose_data_dir = os.path.join(data_dir, 'pose_npy_dir_avi')
files = os.listdir(pose_data_dir)
for file in tqdm(files):
  filename = file.split('.')[0]
  avatar_data = np.load(os.path.join(avatar_data_dir, filename + '_raw.npy'))
  pose_data = np.load(os.path.join(pose_data_dir, filename + '.npy'))
  pose_data = pose_data[:,:,:avatar_data.shape[0]] # Match the frames
  num_frames = pose_data.shape[2]
  for frame_idx in range(num_frames - num_intermediate_frames - 1):
    start_frame = torch.from_numpy(pose_data[:,:,frame_idx]).float()
    end_frame = torch.from_numpy(pose_data[:,:,frame_idx+num_intermediate_frames+1]).float()
    x = torch.cat([start_frame.unsqueeze(0), end_frame.unsqueeze(0)], 0)
    intermediate_frames = torch.from_numpy(pose_data[:,:,frame_idx+1:frame_idx+num_intermediate_frames+1]).float()
    y = intermediate_frames.permute(2,0,1)
    dataset_x.append(x.unsqueeze(0))
    dataset_y.append(y.unsqueeze(0))
  # break
dataset_x = torch.cat(dataset_x, 0)
dataset_y = torch.cat(dataset_y, 0)

interpolation_dataset = {'x': dataset_x, 'y': dataset_y} 
torch.save(interpolation_dataset, 'interpolation.dataset')

print(dataset_x[0].size(), dataset_y[0].size())
