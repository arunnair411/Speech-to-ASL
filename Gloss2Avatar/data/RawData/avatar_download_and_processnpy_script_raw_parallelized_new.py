import wget
import os, sys
from openpyxl import load_workbook

import pandas as pd
import torch
import numpy as np
import skvideo.io
from tqdm import tqdm
import cv2
import multiprocessing
from multiprocessing import Pool
PROCESSES = multiprocessing.cpu_count()//2

import pdb

def get_hyperlink(s):
  if s[:10] == "=HYPERLINK":
    v = s.split(",")[0][12:-1]
    return v
  else:
    return s

def crop_video_parallelized(video, storage_filename, start_frame, end_frame):
  videodata = skvideo.io.vread(video)
  # Original video is of size 640x480 - making it 480x480
  videodata = videodata[start_frame:end_frame+1, :, 80:-80, :] # Sometimes arms are missed, so expanding it
  for idx in range(videodata.shape[0]):
    img = videodata[idx,:,:,:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
    videodata[idx,:,:,:] = cv2.inpaint(img, mask, 17, cv2.INPAINT_NS)
  videodata[:, :100, 0:10, :] = videodata[:, :100, 10:20, :]
  assert videodata.shape[0] == end_frame - start_frame + 1
  np.save(os.path.join('npy_dir', storage_filename.replace('.mov', '_raw.npy')), videodata)
  return 0

def crop_video_parallelized_batched(video_list, storage_filename_list, start_frame_list, end_frame_list, process_number):
  if process_number==0:
    with tqdm(total=len(video_list), file=sys.stdout) as pbar:
      for idx in range(len(video_list)):
        video = video_list[idx]
        storage_filename = storage_filename_list[idx]
        start_frame = start_frame_list[idx]
        end_frame = end_frame_list[idx]
        _ = crop_video_parallelized(video, storage_filename, start_frame, end_frame)
        pbar.update(1)
  else:
    for idx in range(len(video_list)):
      video = video_list[idx]
      storage_filename = storage_filename_list[idx]
      start_frame = start_frame_list[idx]
      end_frame = end_frame_list[idx]
      _ = crop_video_parallelized(video, storage_filename, start_frame, end_frame)
  return 0

def main():
  # !wget http://www.bu.edu/asllrp/dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx
  xlsx_file_name = 'dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx'
  if not os.path.exists(xlsx_file_name):
    wget.download('http://www.bu.edu/asllrp/'+xlsx_file_name, xlsx_file_name)
  wb = load_workbook(filename=xlsx_file_name)
  ws = wb.active

  for row in ws.rows:
      for cell in row:
          try:
              cell.value = get_hyperlink(cell.value)
          except:
              pass
  wb.save("data.xlsx")    

  data = pd.read_excel("data.xlsx")
  data.columns = data.columns.str.replace(' ', '_')
  data = data[data['Consultant'] == 'Liz']
  # data = data[:3]
  data

  raw_filename_list = []
  storage_filename_list = []
  gloss_list = []
  gloss_variant_list = []
  d_start_hs_list = []
  nd_start_hs_list = []
  d_end_hs_list = []
  nd_end_hs_list = []
  passive_arm_list = []
  start_list = []
  end_list = []
  with tqdm(total=len(data), file=sys.stdout) as pbar:
    for index, row in data.iterrows():
      gloss = row['Main_New_Gloss.1']
      gloss_variant = row['Gloss_Variant']
      d_start_hs = row['D_Start_HS']
      nd_start_hs = row['N-D_Start_HS']
      d_end_hs = row['D_End_HS']
      nd_end_hs = row['N-D_End_HS']
      passive_arm = row['Passive_Arm']
      # url = row['Separate']
      camera_number = 1
      url = f"http://csr.bu.edu/ftp/asl/asllvd/asl-data2/quicktime/{row['Session']}/scene{row['Scene']}-camera{camera_number}.mov"
      raw_filename = url[url.find('scene'):]
      url_for_storage_filename = row['Separate']
      storage_filename = url_for_storage_filename[url_for_storage_filename.find('Liz'):]
      # !wget $url # os.system('wget $url')
      os.makedirs(os.path.join('original_mov_dir', row['Session']), exist_ok=True)
      raw_filename_full_path = os.path.join('original_mov_dir', row['Session'], raw_filename)
      if not os.path.exists(raw_filename_full_path):
        wget.download(url, raw_filename_full_path)
      start_list.append(row['Start'])
      end_list.append(row['End'])
      gloss_list.append(gloss)
      gloss_variant_list.append(gloss_variant)
      d_start_hs_list.append(d_start_hs)
      nd_start_hs_list.append(nd_start_hs)
      d_end_hs_list.append(d_end_hs)
      nd_end_hs_list.append(nd_end_hs)
      passive_arm_list.append(passive_arm)
      raw_filename_list.append(raw_filename_full_path)
      storage_filename_list.append(storage_filename)
      # videodata_list.append(videodata)
      pbar.update(1)

  output_files = {'gloss': gloss_list, 'gloss_variant': gloss_variant_list, 'd_start_hs': d_start_hs_list,
                  'nd_start_hs': nd_start_hs_list, 'd_end_hs':d_end_hs_list, 'nd_end_hs' : nd_end_hs_list, 
                  'passive_arm': passive_arm_list, 'filename': storage_filename_list}
  torch.save(output_files, "Liz.data")
  
  os.makedirs('npy_dir', exist_ok=True)

  # # Serial code
  # for i in range(len(raw_filename_list)):
  #   crop_video_parallelized(raw_filename_list[i], storage_filename_list[i], start_list[i], end_list[i])
    
  # # Parallel code - individual threads for individual files
  # multi_pool = multiprocessing.Pool(processes=PROCESSES)
  # _ = multi_pool.starmap(crop_video_parallelized, zip(raw_filename_list, storage_filename_list, start_list, end_list))

  # Parallel code - a thread for an entire batch of files to deal with overheads of assigning to a batch
  # Code below splits files into #PROCESSES batches,wproximately the same number of files for each
  files_per_thread = len(raw_filename_list)//PROCESSES
  raw_filename_list_batched = []
  storage_filename_list_batched = []
  start_list_batched = []
  end_list_batched = []
  for idx in range(PROCESSES-1):
    raw_filename_list_batched.append(raw_filename_list[idx*files_per_thread:(idx+1)*files_per_thread])
    storage_filename_list_batched.append(storage_filename_list[idx*files_per_thread:(idx+1)*files_per_thread])
    start_list_batched.append(start_list[idx*files_per_thread:(idx+1)*files_per_thread])
    end_list_batched.append(end_list[idx*files_per_thread:(idx+1)*files_per_thread])
  idx = PROCESSES-1
  raw_filename_list_batched.append(raw_filename_list[idx*files_per_thread:])
  storage_filename_list_batched.append(storage_filename_list[idx*files_per_thread:])
  start_list_batched.append(start_list[idx*files_per_thread:])
  end_list_batched.append(end_list[idx*files_per_thread:])
  
  multi_pool = multiprocessing.Pool(processes=PROCESSES)
  _ = multi_pool.starmap(crop_video_parallelized_batched, zip(raw_filename_list_batched, storage_filename_list_batched, start_list_batched, end_list_batched, range(PROCESSES)))

  # Close the parallel pool
  multi_pool.close()
  multi_pool.join()

if __name__ == '__main__':
    main()
