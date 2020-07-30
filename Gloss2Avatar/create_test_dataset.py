import torch
from tqdm import tqdm
from skimage.draw import disk
import os
import numpy as np
import h5py
import pdb
import cv2
import torchvision
import argparse

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-height-resized', type=int, default='256')
    parser.add_argument('--img-width-resized', type=int, default='256')
    parser.add_argument('--output-dir', type=str, default='pose2avatar_data_256')
    parser.add_argument('--file-ids', type=str, default='207,15602,15316', help='comma separated ids of gloss files (i.e. Liz_***_raw.npy) to load')
    parser.add_argument('--pose-data', type=str, default='car_she_drives.posesequence')

    parsed_args = parser.parse_args()

    dataset_pose = []
    dataset_avatar = []

    num_pose_keypoints = 25
    num_face_keypoints = 70
    num_hand_keypoints = 21*2

    img_height = 480
    img_width  = 480
    img_height_resized = parsed_args.img_height_resized
    img_width_resized  = parsed_args.img_width_resized
    confidence_threshold = 0.1 ## ADJUST AS REQUIRED - setting it low since we are concerned more with false positives than false negatives - the network can (hopefully) learn to denoise noisy poses

    file_ids = parsed_args.file_ids.split(',')
    # Load pose data
    pose_data = torch.load(parsed_args.pose_data).detach().cpu().numpy()
    # Load avatar data
    num_output_frames = pose_data.shape[0]
    avatar_data = np.zeros((pose_data.shape[0], img_height, img_width, 3))
    # The files this is based on are Liz_207, Liz_15602, Liz_15316, in that order
    # Each has 31 frames -> (20 frame transition) -> 49 frames -> (20 frames transition) -> 41 frames
    # predicted_points has 161 frames (matches with the above)
    tran_length = 20 # in number of frames
    num_files = len(file_ids)
    file_data = []
    file_lens = []
    for idx in range(num_files):
        file_data.append(np.load(f'data/RawData/npy_dir/Liz_{file_ids[idx]}_raw.npy'))
        file_lens.append(file_data[idx].shape[0])

    for idx in range(num_files):
        if idx==0:
            avatar_data[:file_lens[idx],:,:,:] = file_data[idx][:,:,:,:]
        else:
            start_idx = np.array(file_lens[0:idx]).sum() + tran_length*(idx-1)
            end_idx = np.array(file_lens[0:idx+1]).sum() + tran_length*(idx-1)
            avatar_data[start_idx:end_idx] = file_data[idx][:,:,:,:]
            avatar_data[start_idx-tran_length:start_idx,:,:,:] = file_data[idx][0,:,:,:]

    output_dir = parsed_args.output_dir
    subfolder = 'test'

    os.makedirs(output_dir, exist_ok=True)
    for A_or_B in ['A', 'B']:
        # for train_val_test in ['train', 'val', 'test']:
        for train_val_test in [ 'test']:
            os.makedirs(os.path.join(output_dir, A_or_B, train_val_test), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    for frame_idx in tqdm(range(pose_data.shape[0])):
        store_file_idx = frame_idx
        pose_img = torch.from_numpy(pose_data[frame_idx,:,:]).unsqueeze(0).float()
        keypoints_visualized = visualize_keypoints(pose_img)
        if not (img_height_resized==img_height and img_width_resized==img_width):
            keypoints_visualized = cv2.resize(keypoints_visualized[0], (img_height_resized, img_width_resized)).astype(np.float32)    
        keypoints_visualized = torch.from_numpy(keypoints_visualized).permute(2,0,1)
        
        avatar_img = avatar_data[frame_idx,:,:,:]
        if not(img_height_resized==img_height and img_width_resized==img_width):
            avatar_img = cv2.resize(avatar_img, (img_height_resized, img_width_resized)).astype(np.float32)*1.0/255.0
        avatar_img = torch.from_numpy(avatar_img).permute(2,0,1)
        pos_path = os.path.join(output_dir, 'A', subfolder, '{}.png'.format(store_file_idx))
        ava_path = os.path.join(output_dir, 'B', subfolder, '{}.png'.format(store_file_idx))
        torchvision.utils.save_image(keypoints_visualized, pos_path)
        torchvision.utils.save_image(avatar_img, ava_path)
        
        combined_path = os.path.join(output_dir, subfolder, '{}.png'.format(store_file_idx))
        torchvision.utils.save_image(torch.cat((keypoints_visualized, avatar_img) , -1), combined_path)
