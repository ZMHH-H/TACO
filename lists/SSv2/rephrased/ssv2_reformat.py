import json
import csv

# 定义输入和输出文件路径
input_file = 'SSv2_labels.txt'
input_json_test = 'test_rephrased.json'
input_json_train = 'train_rephrased.json'
output_file = 'SSv2_rephrased.csv'

# 打开输入文件并读取行
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 读取JSON文件内容
with open(input_json_test, 'r', encoding='utf-8') as infile:
    data_test = json.load(infile)

# 读取JSON文件内容
with open(input_json_train, 'r', encoding='utf-8') as infile:
    data_train = json.load(infile)

# 打开输出文件以写入CSV内容
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # 写入表头
    csvwriter.writerow(['id', 'name'])
    count = 0
    # 逐行处理输入文件内容
    for line in lines:
        
        # 使用第一个逗号分割行内容
        id= line.strip().split(',', 1)[0]
        name= line.strip().split(',', 1)[1]

        # 逐个项处理JSON内容
        for key, value in data_test.items():
            # print(value.strip().split(':',1)[0])
            if name==value.strip().split(':',1)[0]:
                csvwriter.writerow([id, value])
                count+=1
        # 逐个项处理JSON内容
        for key, value in data_train.items():
            if name==value.strip().split(':',1)[0]:
                csvwriter.writerow([id, value])
                count+=1
    print('total count:', count)
print("文件处理完成，CSV文件已生成。")
