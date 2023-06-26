import os
import numpy as np
import csv
import math

def sliceData(stripname,root_path,slice_path) :
    Data_path_list = os.listdir(root_path)
    for each_path_org in Data_path_list:
        # 获取文件名
        file_name = each_path_org.strip(stripname)
        # 因为是放在originalData下面的，所以要组合成originalData\\000.obs
        each_path = os.path.join(root_path, each_path_org)
        # 打开文件并把n*n*20（时间点）条数据存到all_info_list数组里
        with open(each_path, 'r') as f:
            lines = f.readlines()
            all_info_list = []
            # 把读出来的string数据，按行和空格分格成list[][]数据
            for line_info_str in lines:
                line_info_list = line_info_str.strip("\n").split(" ")
                all_info_list.append(line_info_list)
        all_length = len(all_info_list)
        nn_length = int(all_length/20)
        nn = int(math.sqrt(nn_length))
        # 把数据切片存成20个 11*11的csv文件
        for waitou in range(20):
            lingdianlist = []
            for litou in range(nn_length):
                index = litou * 20 + waitou
                lingdianlist.append(abs(str_float(all_info_list[index][4])*1000000000000))
            lingdianlistn_n = np.array(lingdianlist).reshape(nn, nn)
            data = lingdianlistn_n / abs(lingdianlistn_n).max()  # 规范化到（-1,1）
            divide_file_name = file_name+str(waitou)+'.csv'
            divide_file_path = os.path.join(slice_path,divide_file_name)
            with open(divide_file_path, 'w', encoding='utf-8', newline='') as fp:
                # 写
                writer = csv.writer(fp)
                # 将数据写入
                writer.writerows(data)

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

sliceData('_originalData.obs','originalData','./data/train/label')#sliceOriginalData
sliceData('_addedNoiseData.obs','addedNoiseData','./data/train/image')




