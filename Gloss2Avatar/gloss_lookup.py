import torch
import argparse

metadata = torch.load('data/RawData/Liz.data')
gloss_list = metadata['gloss']
filename_list = metadata['filename']

# ## EXAMPLE GLOSS SEQUENCES
# # gloss_sequence = ['we','receive','documents']
# gloss_sequence = ['point-to','get', 'paper']
# for g in gloss_sequence:
#     # print(filename_list[find_str(gloss_list, g)])
#     print(filename_list[gloss_list.index(g.upper())])


# gloss_sequence = ['car','point-to','drive']
# for g in gloss_sequence:
#     # print(filename_list[find_str(gloss_list, g)])
#     print(filename_list[gloss_list.index(g.upper())])

# # gloss_sequence = ['we','receive','documents']
# gloss_sequence = ['point-to','get', 'paper']
# for g in gloss_sequence:
#     # print(filename_list[find_str(gloss_list, g)])
#     print(filename_list[gloss_list.index(g.upper())])

# # How are you?
# gloss_sequence = ['how','point-to']
# for g in gloss_sequence:
#     # print(filename_list[find_str(gloss_list, g)])
#     print(filename_list[gloss_list.index(g.upper())])

# # "I forwarded the texts to you."  >>15439,7322,2917
# gloss_sequence = ['texting','go-to-it', 'X']
# for g in gloss_sequence:
#     # print(filename_list[find_str(gloss_list, g)])
#     print(filename_list[gloss_list.index(g.upper())])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--glosses', type=str, default='car,point-to,drive', help="enter comma seprated glosses to lookup corresponding video")
  args = parser.parse_args()

  gloss_sequence = args.glosses.split(',')
for g in gloss_sequence:
    print(filename_list[gloss_list.index(g.upper())])
