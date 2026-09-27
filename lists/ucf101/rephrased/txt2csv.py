import csv

# 定义输入和输出文件路径
input_file = 'ucf101_rephrased_new2.txt'
output_file = 'ucf101_rephrased_2.csv'

# 打开输入文件并读取行
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 打开输出文件以写入CSV内容
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # 写入表头
    csvwriter.writerow(['id', 'name'])

    # 逐行处理输入文件内容
    for line in lines:
        # 使用第一个逗号分割行内容
        parts = line.strip().split(',', 1)
        
        # 确保分割后的内容有两部分
        if len(parts) == 2:
            csvwriter.writerow(parts)
        else:
            print(f"Skipping line: {line.strip()} - Incorrect format")

print("文件处理完成，CSV文件已生成。")
