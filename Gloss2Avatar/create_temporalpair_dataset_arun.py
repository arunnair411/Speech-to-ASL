import torch
from tqdm import tqdm
from skimage.draw import disk
import os
import numpy as np
import h5py
import pdb
import cv2

dataset_pose = []
dataset_avatar = []

num_pose_keypoints = 25
num_face_keypoints = 70
num_hand_keypoints = 21*2

confidence_threshold = 0.1 ## ADJUST AS REQUIRED - setting it low since we are concerned more with false positives than false negatives - the network can (hopefully) learn to denoise noisy poses
img_height = 480
img_width = 480

img_height_resized = 256
img_width_resized = 256

def create_disk(canvas, idx, keypoint_array_pixels, r, keypoint_generator):    
    for jdx in keypoint_generator:
        keypoint_confidence = keypoint_array_pixels[idx, jdx, 2]
        if keypoint_confidence<confidence_threshold:
            continue
        else:
            y = keypoint_array_pixels[idx, jdx, 1]
            x = keypoint_array_pixels[idx, jdx, 0]
            rr, cc = disk((y, x), 5)
            rr[rr<0]=0
            rr[rr>=img_height]=img_height-1
            cc[cc<0]=0
            cc[cc>=img_width]=img_width-1
            canvas[rr, cc] = 1


def visualize_keypoints(keypoint_array):
    # keypoint_array is assumed to be a #frames*137*3 array
    num_output_frames = keypoint_array.shape[0]
    keypoint_array_pixels = np.zeros_like(keypoint_array)
    keypoint_array_pixels[:, :, 0] = np.floor(keypoint_array[:, :, 0] * img_width)
    keypoint_array_pixels[:, :, 1] = np.floor(keypoint_array[:, :, 1] * img_height)
    keypoint_array_pixels[:, :, 2] = keypoint_array[:, :, 2]

    keypoint_video = np.zeros((keypoint_array.shape[0], img_height, img_width, 3))
    keypoint_video = np.transpose(keypoint_video, (0,3,1,2))
    for idx in range(num_output_frames):
        # First, pose keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)                
        create_disk(canvas, idx, keypoint_array_pixels, 7, range(num_pose_keypoints))
        keypoint_video[idx][2][np.where(canvas>0)] = 1.0
        # Second, face keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)
        create_disk(canvas, idx, keypoint_array_pixels, 3, range(25, 25+num_face_keypoints))
        keypoint_video[idx][1][np.where(canvas>0)] = 1.0
        # Third, hand keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)
        create_disk(canvas, idx, keypoint_array_pixels, 3, range(25+70, 25+70+num_hand_keypoints))
        keypoint_video[idx][0][np.where(canvas>0)] = 1.0
    keypoint_video = np.transpose(keypoint_video, (0,2,3,1))
    return keypoint_video

# Load data
data_dir = 'data/RawData/'
avatar_data_dir = os.path.join(data_dir, 'npy_dir')
pose_data_dir = os.path.join(data_dir, 'pose_npy_dir_avi')
files = os.listdir(pose_data_dir)
for file in tqdm(files[:10]):
  filename = file.split('.')[0]
  avatar_data = np.load(os.path.join(avatar_data_dir, filename + '_raw.npy'))
  pose_data = np.load(os.path.join(pose_data_dir, filename + '.npy'))
  pose_data = pose_data[:,:,:avatar_data.shape[0]] # Match the frames
  num_frames = pose_data.shape[2]
  for frame_idx in range(num_frames-1):
    pose_pair = torch.from_numpy(pose_data[:,:,frame_idx:frame_idx+2]).float().permute(2,0,1)    
    keypoints_visualized = visualize_keypoints(pose_pair)
    if img_height_resized==img_height and img_width_resized==img_width:
        keypoints_visualized = np.concatenate((keypoints_visualized[0], keypoints_visualized[1]), axis=-1)
    else:
        keypoints_visualized = np.concatenate((cv2.resize(keypoints_visualized[0], (img_height_resized, img_width_resized)),  
                cv2.resize(keypoints_visualized[1], (img_height_resized, img_width_resized))), axis=-1)
    keypoints_visualized = torch.from_numpy(keypoints_visualized).float()
    
    avatar_pair = avatar_data[frame_idx:frame_idx+2,:,:,:]
    if img_height_resized==img_height and img_width_resized==img_width:
        avatar_pair = np.concatenate((avatar_pair[0], avatar_pair[1]), axis=-1)
    else:
        avatar_pair = np.concatenate((cv2.resize(avatar_pair[0], (img_height_resized, img_width_resized)),  
                cv2.resize(avatar_pair[1], (img_height_resized, img_width_resized))), axis=-1)
    avatar_pair = torch.from_numpy(avatar_pair).float()
    
    # dataset_pose.append(pose_pair.unsqueeze(0))
    dataset_pose.append(keypoints_visualized.unsqueeze(0))
    dataset_avatar.append(avatar_pair.unsqueeze(0))
dataset_pose = torch.cat(dataset_pose, 0)
dataset_avatar = torch.cat(dataset_avatar, 0)

temporalpair_dataset = {'pose_pair': dataset_pose, 'avatar_pair': dataset_avatar} 
torch.save(temporalpair_dataset, '3_temporalpair.dataset')

# use h5py to compress dataset
# output_file = h5py.File(output_filepath, "w", libver='latest')

print(dataset_pose[0].size(), dataset_avatar[0].size())
