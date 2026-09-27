import csv

def read_csv(file_path):
    """读取CSV文件并返回一个列表，其中每个元素是一个字典，包含id和name"""
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]

def read_csv_rephrased(file_path):
    """读取CSV文件并返回一个列表，其中每个元素是一个字典，包含id和name"""
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data=[]
        for row in reader:
            row['name']=row['name'].strip().split(':',1)[0]
            data.append(row)
        return data

def compare_csv(file1, file2):
    """比较两个CSV文件的顺序和内容"""
    data1 = read_csv(file1)
    data2 = read_csv_rephrased(file2)
    
    if len(data1) != len(data2):
        print('dict length:',len(data1),len(data2))
        return False, "行数不同"

    for i, (row1, row2) in enumerate(zip(data1, data2)):
        if row1 != row2:
            return False, f"第 {i+1} 行不同: {row1} != {row2}"
    
    return True, "文件内容和顺序一致"

# 定义输入文件路径
file1 = 'SSv2_labels.csv'
file2 = 'SSv2_rephrased.csv'

# 比较两个文件
are_equal, message = compare_csv(file1, file2)

if are_equal:
    print("两个CSV文件的顺序和内容一致。")
else:
    print(f"两个CSV文件不同: {message}")