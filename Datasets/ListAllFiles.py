"""
{'video_name': ['img_name', ...]}
"""


import os
import json
from tqdm import tqdm

DATA_ROOT = 'F:\\Datasets\\YTVOS\\VOS2019\\train_all_frames\\JPEGImages'

all_dir_list = os.listdir(DATA_ROOT)
all_file_dict = {}

for current_directory in tqdm(all_dir_list):
    file_list = os.listdir(DATA_ROOT + '\\' + current_directory)
    file_list.sort()
    all_file_dict[current_directory] = file_list

with open('train_all_frames_list.json', 'w') as out_file:
    json.dump(all_file_dict, out_file)
