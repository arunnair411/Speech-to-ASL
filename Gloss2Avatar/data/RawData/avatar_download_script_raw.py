import wget
import os, sys
from openpyxl import load_workbook

import pandas as pd
import torch
import numpy as np
import skvideo.io
from tqdm import tqdm

import pdb

# !wget http://www.bu.edu/asllrp/dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx
xlsx_file_name = 'dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx'
if not os.path.exists(xlsx_file_name):
  wget.download('http://www.bu.edu/asllrp/'+xlsx_file_name, xlsx_file_name)


wb = load_workbook(filename=xlsx_file_name)
ws = wb.active
def get_hyperlink(s):
  if s[:10] == "=HYPERLINK":
    v = s.split(",")[0][12:-1]
    return v
  else:
    return s
for row in ws.rows:
    for cell in row:
        try:
            cell.value = get_hyperlink(cell.value)
        except:
            pass
wb.save("data.xlsx")


def crop_video(video, start_frame, end_frame):
  videodata = skvideo.io.vread(video)
  # Original video is of size 640x480 - want to keep 384 frames in width, so keep [128:-128]
  videodata = videodata[start_frame:end_frame+1, 48:, 128:-128, :]
  videodata = np.pad(videodata,((0,0), (48,0), (0,0), (0,0)), mode='edge') # Pad it by replicating edge pixels - easy way to keep padding with grey
  assert videodata.shape[0] == end_frame - start_frame + 1
  return videodata

data = pd.read_excel("data.xlsx")
data.columns = data.columns.str.replace(' ', '_')
data = data[data['Consultant'] == 'Liz']
# data = data[:3]
data

# gloss_list = []
# videodata_list = []
# with tqdm(total=len(data), file=sys.stdout) as pbar:
#   for index, row in data.iterrows():
#     gloss = row['Main_New_Gloss.1']
#     url = row['Separate']
#     filename = url[url.find('Liz'):]
#     # !wget $url # os.system('wget $url')
#     if not os.path.exists(filename):
#       wget.download(url, filename)  

#     videodata = crop_video(filename)
#     gloss_list.append(gloss)
#     videodata_list.append(videodata)
#     pbar.update(1)

# output_files = {'gloss': gloss_list,
#                 'videodata': videodata_list}
# torch.save(output_files, "Liz.data")

filename_list = []
gloss_list = []
gloss_variant_list = []
d_start_hs_list = []
nd_start_hs_list = []
d_end_hs_list = []
nd_end_hs_list = []
passive_arm_list = []
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
    if not os.path.exists(os.path.join('original_mov_dir', row['Session'], raw_filename)):
      wget.download(url, os.path.join('original_mov_dir', row['Session'], raw_filename))
    videodata = crop_video(os.path.join('original_mov_dir', row['Session'], raw_filename), row['Start'], row['End'])
    os.makedirs('npy_dir', exist_ok=True)
    np.save(os.path.join('npy_dir', storage_filename.replace('.mov', '_raw.npy')), videodata)
    gloss_list.append(gloss)
    gloss_variant_list.append(gloss_variant)
    d_start_hs_list.append(d_start_hs)
    nd_start_hs_list.append(nd_start_hs)
    d_end_hs_list.append(d_end_hs)
    nd_end_hs_list.append(nd_end_hs)
    passive_arm_list.append(passive_arm)
    filename_list.append(filename_list)
    # videodata_list.append(videodata)
    pbar.update(1)

output_files = {'gloss': gloss_list, 'gloss_variant': gloss_variant_list, 'd_start_hs': d_start_hs_list,
                'nd_start_hs': nd_start_hs_list, 'd_end_hs':d_end_hs_list, 'nd_end_hs' : nd_end_hs_list, 
                'passive_arm': passive_arm_list, 'filename': filename_list}
torch.save(output_files, "Liz.data")
 
###########################################################################################################################################
# # Convert video into numpy
# # !wget http://csr.bu.edu/ftp/asl/asllvd/demos/verify_start_end_handshape_annotations//test_auto_move//signs_mov_separ_signers/Liz_10.mov
# # %pip install scikit-video

# if not os.path.exists('Liz_10.mov'):
#   wget.download('http://csr.bu.edu/ftp/asl/asllvd/demos/verify_start_end_handshape_annotations//test_auto_move//signs_mov_separ_signers/Liz_10.mov')  

# # from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt
# plt.imshow(crop_video("Liz_10.mov")[4])
# plt.show()
