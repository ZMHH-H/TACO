trainlist_path = './trainlist01-1.txt'
folder_path = './testlist01.txt'
output_path = './testlist01-1.txt'

label_dict = dict()
f1 = open(trainlist_path, 'r')
for line in f1:
    path,label = line.strip('\n').split(' ')
    print('path:',path)
    print('label:',label)
    if path.split('/')[0] not in label_dict.keys():
        label_dict[path.split('/')[0]] = label
f1.close()
print(label_dict)
print(len(label_dict))

video_list = []
f2 = open(folder_path, 'r')
for line in f2:
    print(line.strip('\n'))
    video_list.append(line.strip('\n'))
f2.close()
print(len(video_list))

with open(output_path, 'w') as f3:
    for video in video_list:
        print('video:',video)
        print('label:',label_dict[video.split('/')[0]])
        f3.write(video+' '+label_dict[video.split('/')[0]]+'\n')