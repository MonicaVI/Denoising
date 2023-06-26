import os
import csv
import math

def sliceData(chunk_size,stripname,root_path,slice_path) :
    # 每个输出文件包含的行数，这里设置为20
    # chunk_size = 20
    Data_path_list = os.listdir(root_path)
    for each_path_org in Data_path_list:
        # 获取文件名
        file_name = each_path_org.strip(stripname)
        # 因为是放在originalData下面的，所以要组合成originalData\\000.obs
        each_path = os.path.join(root_path, each_path_org)

        with open(each_path, 'r') as input_file:
            lines = input_file.readlines()
            original_data = []
            # 把读出来的string数据，按行和空格分格成list[][]数据
            for line_info_str in lines:
                line_info_list = line_info_str.strip("\n").split(" ")
                original_data.append(line_info_list)
        # 计算原始数据行数
        num_rows = len(original_data)
        # 每20行保存为一个1行20列的CSV文件
        for i in range(0, num_rows, chunk_size):
            # 创建新的CSV文件路径
            divide_file_name = f'{file_name}{i // chunk_size + 1}.csv'
            output_file = os.path.join(slice_path, divide_file_name)
            # 获取当前20行数据
            data_subset = original_data[i:i + chunk_size]
            timeslist = []
            for index in range (chunk_size):
                # 提取每一行的最后一列数据
                timeslist.append(abs(str_float(data_subset[index][4])*1000000000000))
            # 写入到新的CSV文件
            with open(output_file, 'w', encoding='utf-8', newline='') as fp:
                # 创建一个二维列表，其中每个子列表只包含当前一行的最后一个元素
                rows = [[float(x)] for x in timeslist]
                # 使用zip和map函数将rows列表进行转置，然后存储到transposed_data变量中
                transposed_data = list(map(list, zip(*rows)))
                # 将数据写入
                writer = csv.writer(fp)
                # 将二维列表写入到CSV文件中
                writer.writerows(transposed_data)

def str_float(str_num):
    before_e = float(str_num.split('e')[0])
    sign = str_num.split('e')[1][:1]
    after_e = int(str_num.split('e')[1][1:])

    if sign == '+':
        float_num = before_e * math.pow(10, after_e)
    elif sign == '-':
        float_num = before_e * math.pow(10, -after_e)
    else:
        float_num = None
        print('error: unknown sign')
    return float_num

# sliceData(20,'_originalData.obs','originalData','./data/train/label')#sliceOriginalData
sliceData(20,'_addedNoiseData.obs','addedNoiseData','./data/train/image')
