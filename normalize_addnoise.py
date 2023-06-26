import csv
import glob
import os

import numpy as np
from numpy import genfromtxt


def normalizeAddnoise(orgdata_path,noisdata_path):
    orgdatapath_list = os.listdir(orgdata_path)
    all_patches =[]
    for each_path_org in orgdatapath_list:
        # 获取文件名
        file_name = each_path_org.strip(".csv")
        # 因为是放在originalData下面的，所以要组合成originalData\\000.obs
        each_path = os.path.join(orgdata_path, each_path_org)
        data = genfromtxt(each_path, delimiter=',', skip_header=False)
        # 增加随机噪声
        noise_factor = 0.1
        data = data + noise_factor * np.random.randn(*data.shape)  # 在归一化的基础上加噪声才明显
        all_patches.append(data)
        print("hello")
        nradnoise_file_name = file_name + '.csv'
        nradnoise_file_path = os.path.join(noisdata_path, nradnoise_file_name)
        with open(nradnoise_file_path, 'w', encoding='utf-8', newline='') as fp:
            # 写
            writer = csv.writer(fp)
            # 将数据写入
            writer.writerows(data)


# normalizeAddnoise('./data/train/label','./data/train/image')
normalizeAddnoise('./data/test/','./data/')
