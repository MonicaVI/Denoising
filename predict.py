import numpy as np
import torch
from psnr import psnr
from calculate_snr import calculate_snr
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from matplotlib import pyplot as plt

from model.unet_model import UNet
from numpy import genfromtxt

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    ########获取测试，测试的时候 在干净数据上加噪声然后送去测试的
    noise = 'data/test/noise.csv'
    noise_data = genfromtxt(noise, delimiter=',', skip_header=False)
    clear = 'data/test/org.csv'
    f = genfromtxt(clear, delimiter=',', skip_header=False)
    clear_data = f[0:64, 0:64]

     # 转为batch为1，通道为1，大小为512*512的数组
    noise_data = noise_data.reshape(1, 1, noise_data.shape[0], noise_data.shape[1])
    # 转为tensor
    noist_tensor = torch.from_numpy(noise_data)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = noist_tensor.to(device=device, dtype=torch.float32)
    # 预测
    pred = net(img_tensor)
    # 提取结果
    data_cons = np.array(pred.data.cpu()[0])[0]
    # 定义了一个显示范围clip和vmin、vmax两个变量。clip表示显示范围的大小，负值越大越明显，vmin和vmax则用于设置颜色映射的取值范围。
    clip = 1e-0
    vmin, vmax = -clip, clip
    # Figure
    figsize = (10, 10)  # 设置图形的大小
    # 并创建3x1的子图(axs)。figsize参数用于设置图形的大小；squeeze参数用于确定是否压缩子图数组的空行和列；dpi参数用于设置图像的分辨率。
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, facecolor='w', edgecolor='k',
                            squeeze=False, sharex=True, dpi=100)
    axs = axs.ravel()  # 将axs矩阵展平为一维数组，以便通过索引访问每个子图。
    # 将data_cons和noise_data数组转换为64x64的形状，并在第一个子图中绘制data数组的热力图。
    # 其中，cmap参数指定了使用哪种颜色映射，vmin和vmax参数用于设置颜色映射的取值范围。
    # 这个是原始图像
    data_cons = np.reshape(data_cons, (32, 32))
    noise_data = np.reshape(noise_data, (32, 32))
    axs[0].imshow(clear_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    axs[0].set_title('Clear')
    axs[0].grid(False)
    ##########################

    clip = 1e-0  # 显示范围，负值越大越明显
    axs[1].imshow(noise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    noisy_psnr = psnr(clear_data, noise_data)
    noisy_psnr = round(noisy_psnr, 2)
    # noisy_snr = calculate_snr(clear_data, noise_data)
    # noisy_snr = round(noisy_snr, 2)
    from calculate_snr import calculate_snr

    # axs[1].set_title('Noisy, psnr=' + str(noisy_psnr) + '/n,SNR=' + str(noisy_snr))
    axs[1].set_title('Noisy, psnr=' + str(noisy_psnr))
    axs[1].grid(False)
    ############################

    # 这个是去噪后的图像，信噪比是15.34，因为训练次数还不够多，只训练了5次
    clip = 1e-0  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    axs[2].imshow(data_cons, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    Denoised_psnr = psnr(clear_data, data_cons)
    Denoised_psnr = round(Denoised_psnr, 2)
    Denoised_snr = calculate_snr(clear_data, noise_data)
    Denoised_snr = round(Denoised_snr, 2)
    # axs[2].set_title('Denoised proposed, psnr=' + str(Denoised_psnr) + ',SNR=' + str(Denoised_snr))
    axs[2].set_title('Denoised proposed, psnr=' + str(Denoised_psnr) )
    axs[2].grid(False)

    plt.show()
    # # 遍历素有图片
    # for test_path in tests_path:
    #     # 保存结果名字
    #     save_res_path = test_path.strip('.csv')+ '_res.csv'
    #     # 读取图片
    #
    #     img = genfromtxt(test_path, delimiter=',', skip_header=False)
    #
    #     # 转为batch为1，通道为1，大小为512*512的数组
    #     img = img.reshape(1, 1, img.shape[0], img.shape[1])
    #     # 转为tensor
    #     img_tensor = torch.from_numpy(img)
    #     # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    #     img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    #     # 预测
    #     pred = net(img_tensor)
    #     # 提取结果
    #     pred = np.array(pred.data.cpu()[0])[0]
    #     # 保存图片
    #     with open(save_res_path, 'w', encoding='utf-8', newline='') as fp:
    #         # 写
    #         writer = csv.writer(fp)
    #         # 将数据写入
    #         writer.writerows(pred)
