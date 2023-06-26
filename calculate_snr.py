# -*- coding: utf-8 -*-
import numpy as np
#######################计算SNR
def calculate_snr(data_origin, reconstructed):
    diff = reconstructed - data_origin
    snr = 20 * np.log10(np.linalg.norm(data_origin) / np.linalg.norm(diff))
    return snr