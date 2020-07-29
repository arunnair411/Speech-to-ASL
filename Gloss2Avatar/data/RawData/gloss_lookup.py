import torch
metadata = torch.load('Liz.data')
gloss_list = metadata['gloss']
filename_list = metadata['filename']
def find_str(l, pat):
 for i, s in enumerate(l):
  if pat.lower() in str(s).lower():
   return i
 return -1

gloss_sequence = ['car','she','drive']
for g in gloss_sequence:
  print(filename_list[find_str(gloss_list, g)])

