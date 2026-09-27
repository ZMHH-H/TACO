folder_path = './testlist01.txt'
output_path = './testlist01-1.txt'

video_list = []
f = open(folder_path, 'r')
for line in f:
    print(line.strip('\n').split(' '))
    video_list.append(line.strip('\n').split(' '))
f.close()
print(len(video_list))

with open(output_path, 'w') as f1:
    for video in video_list:
        f1.write(video[0]+' '+str(int(video[1])-1)+'\n')