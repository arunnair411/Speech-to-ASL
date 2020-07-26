import os, glob, json
import numpy as np
import pdb

reconstructed_mov_dir_avi_path = "C:\\Users\\t-arnair\\Desktop\\ASL\\RawData\\reconstructed_mov_dir_avi"
file_names = sorted(glob.glob(os.path.join(reconstructed_mov_dir_avi_path,"*.avi")))
json_dir = "C:\\Users\\t-arnair\\Desktop\\ASL\\RawData\\json_dir"
pose_npy_dir = "C:\\Users\\t-arnair\\Desktop\\ASL\\RawData\\pose_npy_dir"
os.makedirs(json_dir, exist_ok=True)
os.makedirs(pose_npy_dir, exist_ok=True)
os.chdir("C:\\Users\\t-arnair\\Desktop\\openpose-1.6.0-binaries-win64-gpu-python-flir-3d_recommended\\openpose") # Change to root directory of the keypoint estimator
for idx, curr_file in enumerate(file_names):
    pdb.set_trace()
    file_name = os.path.basename(curr_file)
    os.makedirs(os.path.join(json_dir, os.path.splitext(file_name)[0]), exist_ok=True)
    # Enable tracking in the hopes of smoothness and potential higher accuracy (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md) - same runtime
    os.system(f"bin\\OpenPoseDemo.exe --keypoint_scale=3 --tracking 0 --number_people_max 1 --video {curr_file} --face --hand --hand_scale_number 6 --hand_scale_range 0.4 --display=0  --render_pose=0  --write_json={os.path.join(json_dir, os.path.splitext(file_name)[0])}")
    # Disable tracking
    # os.system(f"bin\\OpenPoseDemo.exe --keypoint_scale=3 --number_people_max 1 --video {curr_file} --face --hand --hand_scale_number 6 --hand_scale_range 0.4 --display=0  --render_pose=0  --write_json={os.path.join(json_dir, os.path.splitext(file_name)[0])}")
    pdb.set_trace()
    json_file_names = sorted(glob.glob(os.path.join(json_dir, os.path.splitext(file_name)[0], '*.json')))
    num_frames = len(json_file_names)
    npy_array = np.zeros((25+70+21+21, 3, num_frames)).astype(np.float16) # 25 for pose, 70 for face, 21 for left and, and 21 for right hand
    for jdx, curr_json_file in enumerate(json_file_names):
        f = open(curr_json_file)
        data = json.load(f)
        if len(data['people'])!=1:
            print(f'Only one person not detected!!! File is {curr_json_file}')
        npy_array[:25, :, jdx] = np.reshape(data['people'][0]['pose_keypoints_2d'], (-1,3)).astype(np.float16)
        npy_array[25:25+70, :, jdx] = np.reshape(data['people'][0]['face_keypoints_2d'], (-1,3)).astype(np.float16)
        npy_array[25+70:25+70+21, :, jdx] = np.reshape(data['people'][0]['hand_left_keypoints_2d'], (-1,3)).astype(np.float16)
        npy_array[25+70+21:25+70+21+21, :, jdx] = np.reshape(data['people'][0]['hand_right_keypoints_2d'], (-1,3)).astype(np.float16)
    np.save(os.path.join(pose_npy_dir, file_name.split('_raw_reconstructed')[0]+'.npy'), npy_array)
