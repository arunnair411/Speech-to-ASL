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

# Below commented function just crops in time
def crop_video_parallelized_keypoints(video, storage_filename, start_frame, end_frame):
  videodata = skvideo.io.vread(video)
  videodata = videodata[start_frame:end_frame+1, :, :, :] # Sometimes arms are missed, so expanding it
  assert videodata.shape[0] == end_frame - start_frame + 1
  frame_rate = '60/1' 
  # OUTPUT_FORMAT = 'AVI' # AVI does worse than MOV...
  OUTPUT_FORMAT = 'MOV'
  if OUTPUT_FORMAT == 'MOV':
    writer = skvideo.io.FFmpegWriter(os.path.join('openpose_input_videos', os.path.basename(storage_filename)), outputdict={'-r': frame_rate, '-vcodec': 'libx264', '-vb': '20M'})
  elif OUTPUT_FORMAT == 'AVI':
    writer = skvideo.io.FFmpegWriter(os.path.join('openpose_input_videos_avi', os.path.basename(storage_filename).replace('.mov', '.avi')), outputdict={'-r': frame_rate, '-vcodec': 'libx264', '-vb': '20M'})
  for i in range(videodata.shape[0]):
    writer.writeFrame(videodata[i,:,:,:])
  try:
    writer.close()
  except:
    pdb.set_trace()
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
      pbar.update(1)

  output_files = {'gloss': gloss_list, 'gloss_variant': gloss_variant_list, 'd_start_hs': d_start_hs_list,
                  'nd_start_hs': nd_start_hs_list, 'd_end_hs':d_end_hs_list, 'nd_end_hs' : nd_end_hs_list, 
                  'passive_arm': passive_arm_list, 'filename': storage_filename_list}
  torch.save(output_files, "Liz.data")  
  
  os.makedirs('openpose_input_videos', exist_ok=True)
#   os.makedirs('openpose_input_videos_avi', exist_ok=True)

#   # Serial code
#   pdb.set_trace()
#   for i in range(len(raw_filename_list)):
#     crop_video_parallelized_keypoints(raw_filename_list[i], storage_filename_list[i], start_list[i], end_list[i])
    
  # Parallel code
  multi_pool = multiprocessing.Pool(processes=PROCESSES)
  _ = multi_pool.starmap(crop_video_parallelized_keypoints, zip(raw_filename_list, storage_filename_list, start_list, end_list))

  # Close the parallel pool
  multi_pool.close()
  multi_pool.join()

if __name__ == '__main__':
    main()
