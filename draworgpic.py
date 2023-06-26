import numpy as np
import torch
from psnr import psnr
from calculate_snr import calculate_snr
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from matplotlib import pyplot as plt
from numpy import genfromtxt

if __name__ == "__main__":
    # 读取所有图片路径
    clear = 'data/test/org.csv'
    f = genfromtxt(clear, delimiter=',', skip_header=False)
    clear_data = f[0:64, 0:64]

    # 定义了一个显示范围clip和vmin、vmax两个变量。clip表示显示范围的大小，负值越大越明显，vmin和vmax则用于设置颜色映射的取值范围。
    clip = 1e-0
    vmin, vmax = -clip, clip
    # Figure
    figsize = (20, 20)  # 设置图形的大小
    # 并创建3x1的子图(axs)。figsize参数用于设置图形的大小；squeeze参数用于确定是否压缩子图数组的空行和列；dpi参数用于设置图像的分辨率。
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w', edgecolor='k',
                            squeeze=False, sharex=True, dpi=100)
    axs = axs.ravel()  # 将axs矩阵展平为一维数组，以便通过索引访问每个子图。
    axs[0].imshow(clear_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    axs[0].set_title('Clear')
    axs[0].grid(False)


    plt.show()

