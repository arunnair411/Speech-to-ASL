import wget
import os, sys
from openpyxl import load_workbook

import pandas as pd
import torch
import numpy as np
import skvideo.io
from tqdm import tqdm
import glob
from itertools import repeat
import multiprocessing
from multiprocessing import Pool
PROCESSES = multiprocessing.cpu_count()//2

import pdb

def save_video_parallelized(file_name, frame_rate):
    # OUTPUT_FORMAT = 'AVI' # AVI does worse than MOV...
    OUTPUT_FORMAT = 'MOV'
    videodata = np.load(file_name)
    if OUTPUT_FORMAT == 'MOV':
        writer = skvideo.io.FFmpegWriter(os.path.join('reconstructed_mov_dir', os.path.basename(file_name).replace('.npy', '_reconstructed.mov')), outputdict={'-r': frame_rate, '-vcodec': 'libx264', '-vb': '20M'})
    elif OUTPUT_FORMAT == 'AVI':
        writer = skvideo.io.FFmpegWriter(os.path.join('reconstructed_mov_dir_avi', os.path.basename(file_name).replace('.npy', '_reconstructed.avi')), outputdict={'-r': frame_rate, '-vcodec': 'libx264', '-vb': '20M'})
    for i in range(videodata.shape[0]):
        writer.writeFrame(videodata[i,:,:,:])
    try:
        writer.close()
    except:
        pdb.set_trace()

def main():
    # assert metadata['video']['@r_frame_rate'].split('/')[0]=='60', 'Non-standard frame rate video'      # Numerical value of framerate # ALWAYS 60, it's cool
    # frame_rate = metadata['video']['@r_frame_rate']
    frame_rate = '60/1'
    file_names = sorted(glob.glob(os.path.join('npy_dir', '*.npy')))
    os.makedirs('reconstructed_mov_dir', exist_ok = True)
    os.makedirs('reconstructed_mov_dir_avi', exist_ok = True)
    # # SERIAL
    # pdb.set_trace()
    # for idx, file_name in enumerate(file_names):
    #     save_video_parallelized(file_name, frame_rate)

    # PARALLEL
    multi_pool = multiprocessing.Pool(processes=PROCESSES)
    _ = multi_pool.starmap(save_video_parallelized, zip(file_names, repeat(frame_rate)))
    # Close the parallel pool
    multi_pool.close()
    multi_pool.join()

if __name__ == '__main__':
    main()
