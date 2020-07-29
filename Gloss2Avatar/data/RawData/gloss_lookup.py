import torch
import pdb
metadata = torch.load('Liz.data')
gloss_list = metadata['gloss']
filename_list = metadata['filename']
def find_str(l, pat):
 for i, s in enumerate(l):
  if pat.lower() in str(s).lower():
   return i
 return -1

# gloss_sequence = ['we','receive','documents']
gloss_sequence = ['point-to','get', 'paper']
for g in gloss_sequence:
    # print(filename_list[find_str(gloss_list, g)])
    print(filename_list[gloss_list.index(g.upper())])


gloss_sequence = ['car','point-to','drives']
for g in gloss_sequence:
    # print(filename_list[find_str(gloss_list, g)])
    print(filename_list[gloss_list.index(g.upper())])

# gloss_sequence = ['we','receive','documents']
gloss_sequence = ['point-to','get', 'paper']
for g in gloss_sequence:
    # print(filename_list[find_str(gloss_list, g)])
    print(filename_list[gloss_list.index(g.upper())])

# How are you?
gloss_sequence = ['how','point-to']
for g in gloss_sequence:
    # print(filename_list[find_str(gloss_list, g)])
    print(filename_list[gloss_list.index(g.upper())])

# "I forwarded the texts to you."  >>15439,7322,2917
gloss_sequence = ['texting','go-to-it', 'X']
for g in gloss_sequence:
    # print(filename_list[find_str(gloss_list, g)])
    print(filename_list[gloss_list.index(g.upper())])
