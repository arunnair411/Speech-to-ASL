import torch
from tqdm import tqdm
import os
import numpy as np
import h5py
from torch.nn import functional as F

dataset_pose = []
dataset_avatar = []

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
  for frame_idx in range(num_frames-1):
    pose_pair = torch.from_numpy(pose_data[:,:,frame_idx:frame_idx+2]).float().permute(2,0,1)
    avatar_pair = torch.from_numpy(avatar_data[frame_idx:frame_idx+2,:,:,:]).float()
    avatar_pair = avatar_pair.permute(0,3,1,2) # 2,3,480,480
    avatar_pair = F.avg_pool2d(avatar_pair, 4, 4)
    dataset_pose.append(pose_pair.unsqueeze(0))
    dataset_avatar.append(avatar_pair.unsqueeze(0))
  # break
dataset_pose = torch.cat(dataset_pose, 0)
dataset_avatar = torch.cat(dataset_avatar, 0)

temporalpair_dataset = {'pose_pair': dataset_pose, 'avatar_pair': dataset_avatar} 
torch.save(temporalpair_dataset, 'temporalpair.dataset')

# use h5py to compress dataset
# output_file = h5py.File(output_filepath, "w", libver='latest')

print(dataset_pose[0].size(), dataset_avatar[0].size())
