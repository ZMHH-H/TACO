'''
relative path
'''

import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import uuid
import numpy as np
import pandas as pd

# folder_path = './k400/val/'
# output_path = './k400/vallist.txt'

file_src = './train_rgb.txt'
output_path = './train.txt'

file_list = []
cls_set=set()
f = open(file_src, 'r')
for line in f:
    # rows = line.split()
    # fname = rows[0]
    # print(line.strip('\n'))
    file_list.append(line.strip('\n'))
print(len(file_list))

with open(output_path,'w') as f:
    for line in file_list:
        f.write(line.split(' ')[0]+'.webm '+line.split(' ')[-1]+'\n')