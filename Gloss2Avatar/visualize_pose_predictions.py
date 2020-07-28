import torch
import numpy as np
from skimage.draw import disk
import skvideo.io
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt

img_width  = 480 # in pixels
img_height = 480 # in pixels

num_pose_keypoints = 25
num_face_keypoints = 70
num_hand_keypoints = 21*2

def create_disk(canvas, idx, keypoint_array_pixels, r, keypoint_generator):
    for jdx in keypoint_generator:
        y = keypoint_array_pixels[idx, jdx, 1]
        x = keypoint_array_pixels[idx, jdx, 0]
        rr, cc = disk((y, x), 5)
        rr[rr<0]=0
        rr[rr>=img_height]=img_height-1
        cc[cc<0]=0
        cc[cc>=img_width]=img_width-1
        canvas[rr, cc] = 255


def visualize_keypoints(keypoint_array):
    # keypoint_array is assumed to be a 141*137*3 array
    num_output_frames = keypoint_array.shape[0]
    keypoint_array_pixels = np.zeros_like(keypoint_array)
    keypoint_array_pixels[:, :, 0] = np.floor(keypoint_array[:, :, 0] * img_width)
    keypoint_array_pixels[:, :, 1] = np.floor(keypoint_array[:, :, 1] * img_height)

    keypoint_video = np.zeros((keypoint_array.shape[0], img_height, img_width, 3))
    keypoint_video = np.transpose(keypoint_video, (0,3,1,2))
    for idx in tqdm(range(num_output_frames)):        
        # First, pose keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)                
        create_disk(canvas, idx, keypoint_array_pixels, 5, range(num_pose_keypoints))
        keypoint_video[idx][2][np.where(canvas>0)] = canvas[np.where(canvas>0)]
        # Second, face keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)
        create_disk(canvas, idx, keypoint_array_pixels, 3, range(25, 25+num_face_keypoints))
        keypoint_video[idx][1][np.where(canvas>0)] = canvas[np.where(canvas>0)]
        # Third, hand keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)
        create_disk(canvas, idx, keypoint_array_pixels, 3, range(25+70, 25+70+num_hand_keypoints))
        keypoint_video[idx][0][np.where(canvas>0)] = canvas[np.where(canvas>0)]
    
    # Write it to a video
    keypoint_video = np.transpose(keypoint_video, (0,2,3,1))
    frame_rate = '5/1'         
    OUTPUT_FORMAT = 'AVI' # AVI does worse than MOV...    
    keypoint_video = keypoint_video.astype(np.uint8)
    writer = skvideo.io.FFmpegWriter('keypoints_visualized.avi', outputdict={'-r': frame_rate, '-vcodec': 'libx264', '-vb': '20M'})
    for i in range(keypoint_video.shape[0]):
        writer.writeFrame(keypoint_video[i,:,:,:])
    try:
        writer.close()
    except:
        pdb.set_trace()
    
    return 0

def visualize_keypoints_on_video(keypoint_array):
    # keypoint_array is assumed to be a 141*137*3 array
    num_output_frames = keypoint_array.shape[0]
    keypoint_array_pixels = np.zeros_like(keypoint_array)
    keypoint_array_pixels[:, :, 0] = np.floor(keypoint_array[:, :, 0] * img_width)
    keypoint_array_pixels[:, :, 1] = np.floor(keypoint_array[:, :, 1] * img_height)

    keypoint_video = np.zeros((keypoint_array.shape[0], img_height, img_width, 3))
    # The files this is based on are Liz_207, Liz_15602, Liz_15316, in that order
    # Each has 31 frames -> (20 frame transition) -> 49 frames -> (20 frames transition) -> 41 frames
    # predicted_points has 161 frames (matches with the above)
    tran_length = 20 # in number of frames
    keypoint_video[:31,:,:,:] = np.load('data/RawData/npy_dir/Liz_207_raw.npy')
    keypoint_video[31+tran_length:31+tran_length+49,:,:,:] = np.load('data/RawData/npy_dir/Liz_15602_raw.npy')
    keypoint_video[31+tran_length+49+tran_length:31+tran_length+49+tran_length+41,:,:,:] = np.load('data/RawData/npy_dir/Liz_15316_raw.npy')
    # Fill in gaps by copying the end frame of the previous pose 5 times and start frame of next pose 5 times
    keypoint_video[31:31+tran_length//2,:,:,:] = keypoint_video[31-1,:,:,:]
    keypoint_video[31+tran_length+49:31+tran_length+49+tran_length//2,:,:,:] = keypoint_video[31+tran_length+49-1,:,:,:]
    keypoint_video[31+tran_length//2:31+tran_length,:,:,:] = keypoint_video[31+tran_length,:,:,:]
    keypoint_video[31+tran_length+49+tran_length//2:31+tran_length+49+tran_length,:,:,:] = keypoint_video[31+tran_length+49+tran_length,:,:,:]
    keypoint_video = np.transpose(keypoint_video, (0,3,1,2))
    for idx in tqdm(range(num_output_frames)):        
        # First, pose keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)
        create_disk(canvas, idx, keypoint_array_pixels, 5, range(num_pose_keypoints))
        keypoint_video[idx][2][np.where(canvas>0)] = canvas[np.where(canvas>0)]
        # Second, face keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)
        create_disk(canvas, idx, keypoint_array_pixels, 3, range(25, 25+num_face_keypoints))
        keypoint_video[idx][1][np.where(canvas>0)] = canvas[np.where(canvas>0)]
        # Third, hand keypoints
        canvas = np.zeros((img_height, img_width), dtype=np.float)
        create_disk(canvas, idx, keypoint_array_pixels, 3, range(25+70, 25+70+num_hand_keypoints))
        keypoint_video[idx][0][np.where(canvas>0)] = canvas[np.where(canvas>0)]
    
    # Write it to a video
    keypoint_video = np.transpose(keypoint_video, (0,2,3,1))
    frame_rate = '5/1'         
    OUTPUT_FORMAT = 'AVI' # AVI does worse than MOV...    
    keypoint_video = keypoint_video.astype(np.uint8)
    writer = skvideo.io.FFmpegWriter('keypoints_visualized_over_video.avi', outputdict={'-r': frame_rate, '-vcodec': 'libx264', '-vb': '20M'})
    for i in range(keypoint_video.shape[0]):
        writer.writeFrame(keypoint_video[i,:,:,:])
    try:
        writer.close()
    except:
        pdb.set_trace()
    
    return 0

if __name__ == '__main__':
    predicted_points = torch.load('car_she_drives.posesequence').detach().cpu().numpy()
    visualize_keypoints(predicted_points)
    visualize_keypoints_on_video(predicted_points)
