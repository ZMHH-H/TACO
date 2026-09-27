import json
import csv

# 定义输入和输出文件路径
input_file = 'k600_split3_rephrased_classes.json'
output_file = 'k160_test_s3_rephrased.csv'

# 读取JSON文件内容
with open(input_file, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 打开输出文件以写入CSV内容
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # 写入表头
    csvwriter.writerow(['id', 'name'])
    
    # 逐个项处理JSON内容
    for key, value in data.items():
        csvwriter.writerow([key, value])

print("文件处理完成，CSV文件已生成。")
