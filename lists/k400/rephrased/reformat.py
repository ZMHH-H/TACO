import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess

file_src = './k400_rephrased.txt'
output_path = './k400_rephrased_new.txt'
class_list=[]
f = open(file_src, 'r')
for line in f:
    if line !='\n':
        class_list.append(line)
print(class_list)
print(len(class_list))
with open(output_path,'w') as f:
    for cls_name in class_list:
        f.write(cls_name)
#     for key in cls_dict.keys():
#         if key in cls_set:
#             print('find corresponding cls name: ',key)
#             count+=1
#             for cls_name in file_list:
#                 cls_info = cls_name.split('/')
#                 if cls_info[0]==key:
#                     f.write(cls_info[0]+'/'+cls_info[-1]+' '+ cls_dict[key] +'\n')