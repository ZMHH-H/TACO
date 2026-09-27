# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import uuid

from joblib import delayed
from joblib import Parallel
import pandas as pd

'''
The labels of ucf101 dataset in official train split txt file
is start from 1 to 101.
But pytorch needs the labels smaller than the number of classes,
so relabel(original label-1) is needed to generate new train file
'''


test_file_src = './hmdb_full_for_zeroshot.txt'
save_file_srz = './hmdb_full_for_zeroshot_1.txt'

# 按行读取原始testlist.txt文件
with open(test_file_src,'r') as f:
    lines = f.readlines()
    print(len(lines))
# 判断每行的内容对应的class index并将内容与class index写入新文件
with open(save_file_srz,'w') as f:
    for line in lines:
        rows = line.strip('\n').split(' ')
        path = rows[0]
        label = rows[2]
        f.write(path+'.avi'+' '+label+'\n')




