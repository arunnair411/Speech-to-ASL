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


with tqdm(total=len(data), file=sys.stdout) as pbar:
    for index, row in data.iterrows():
        camera_number=1
        url = f"http://csr.bu.edu/ftp/asl/asllvd/asl-data2/quicktime/{row['Session']}/scene{row['Scene']}-camera{camera_number}.mov"
        filename = url[url.find('scene'):]
        metadata = skvideo.io.ffprobe(os.path.join('original_mov_dir', row['Session'],  filename))
        assert metadata['video']['@r_frame_rate'].split('/')[0]=='60', 'Non-standard frame rate video'      # Numerical value of framerate
        frame_rate = metadata['video']['@r_frame_rate']

        url = row['Separate']
        filename = url[url.find('Liz'):]
        videodata = np.load(os.path.join('npy_dir', filename.replace('.mov', '_raw.npy')))

        os.makedirs('reconstructed_mov_dir', exist_ok = True)
        writer = skvideo.io.FFmpegWriter(os.path.join('reconstructed_mov_dir', filename.replace('.mov', '_raw_reconstructed.mov')), outputdict={'-r': frame_rate})
        for i in range(videodata.shape[0]):
            writer.writeFrame(videodata[i,:,:,:])
        try:
            writer.close()
        except:
            pdb.set_trace()
        pbar.update(1)

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