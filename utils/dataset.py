import torch
import os
import glob
from torch.utils.data import Dataset
from numpy import genfromtxt
from pathlib import Path



class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = Path(data_path)
        self.imgs_path = glob.glob(os.path.join(data_path, '*.csv'))

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = genfromtxt(image_path, delimiter=',', skip_header=False)
        label = genfromtxt(label_path, delimiter=',', skip_header=False)
        # 图的时候用这个
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 时序时候用下面这个
        # image = image.reshape(1, 1, image.shape[0])
        # label = label.reshape(1, 1, label.shape[0])
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("E:\\Project\\Denoising\\data\\train\\image")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(label.shape)