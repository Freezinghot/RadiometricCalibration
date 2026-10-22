# -*- coding: utf-8 -*-
# @File  : split_bin2raw.py
# @Author: Freezinghot
# @Date  : 2026/7/23
# @Desc  :
import os
import numpy as np
import tifffile
from tqdm import tqdm


def write_tiff(data, out_filename):
    tifffile.imwrite(out_filename, data, photometric='minisblack')

def mss_split_bin2raw(mss_binname, output_dir):
    basename = os.path.basename(mss_binname)
    b1_filename = os.path.join(output_dir, 'B1_'+basename.replace('bin', 'tif'))
    b2_filename = os.path.join(output_dir, 'B2_'+basename.replace('bin', 'tif'))
    b3_filename = os.path.join(output_dir, 'B3_'+basename.replace('bin', 'tif'))
    b4_filename = os.path.join(output_dir, 'B4_'+basename.replace('bin', 'tif'))
    with open(mss_binname, 'rb') as bf:
        bf_data = bf.read()
    n = 4544
    bf_split = [bf_data[i:i+n] for i in range(0, len(bf_data), n)]
    groups = tuple([bf_split[i] for i in range(start, len(bf_split), 4)] for start in range(4))
    for bs in groups:
        if bs[0][10] == 1:
            trimmed = [d[256:-4] for d in bs]
            data_bytes = b''.join(trimmed)
            unpack_bytes = convert_16bit_to_12bit(data_bytes)
            tifffile.imwrite(b1_filename, unpack_bytes, photometric='minisblack')
        if bs[0][10] == 2:
            trimmed = [d[256:-4] for d in bs]
            data_bytes = b''.join(trimmed)
            unpack_bytes = convert_16bit_to_12bit(data_bytes)
            tifffile.imwrite(b2_filename, unpack_bytes, photometric='minisblack')
        if bs[0][10] == 3:
            trimmed = [d[256:-4] for d in bs]
            data_bytes = b''.join(trimmed)
            unpack_bytes = convert_16bit_to_12bit(data_bytes)
            tifffile.imwrite(b3_filename, unpack_bytes, photometric='minisblack')
        if bs[0][10] == 4:
            trimmed = [d[256:-4] for d in bs]
            data_bytes = b''.join(trimmed)
            unpack_bytes = convert_16bit_to_12bit(data_bytes)
            tifffile.imwrite(b4_filename, unpack_bytes, photometric='minisblack')

def pan_bin2raw(pan_filename, output_dir):
    basename = os.path.basename(pan_filename)
    export_filename = os.path.join(output_dir, 'P_' + basename.replace('bin', 'tif'))
    with open(pan_filename, 'rb') as bf:
        bf_data = bf.read()
    n = 17392       # (8568+128) * 2
    pf_split = [bf_data[i:i+n] for i in range(0, len(bf_data), n)]
    for pf in pf_split:
        trimmed = [d[256:] for d in pf]
        data_bytes = b''.join(trimmed)
        unpack_bytes = convert_16bit_to_12bit(data_bytes)
        tifffile.imwrite(output_dir, unpack_bytes, photometric='minisblack')


def convert_16bit_to_12bit(data_bytes, mode='low', endian='little', width=2142):
    """
    批量转换2字节数据为12位值，返回uint16数组

    Args:
        data_bytes: bytes对象
        mode: 'low' - 数据在低12位; 'shift' - 数据左移4位
        endian: 'little' 或 'big'

    Returns:
        np.ndarray: uint16类型的12位值数组 (0-4095)
    """
    if len(data_bytes) % 2 != 0:
        raise ValueError("数据长度必须是2的倍数")

    # 将bytes转换为uint16数组（根据端序）
    if endian == 'little':
        dtype = np.dtype('<u2')  # 小端序 uint16
    else:
        dtype = np.dtype('>u2')  # 大端序 uint16

    # 直接转换为uint16数组
    values_16bit = np.frombuffer(data_bytes, dtype=dtype)

    # 提取12位值
    if mode == 'low':
        values_12bit = values_16bit & 0xFFF
    elif mode == 'shift':
        values_12bit = values_16bit >> 4
    else:
        raise ValueError("mode必须是 'low' 或 'shift'")

    height = len(values_12bit)//width
    return values_12bit.astype(np.uint16).reshape(height, width)


def batch_extract_mss(mss_binfiles, export_dir):
    for i in tqdm(range(len(mss_binfiles)), desc='Extract MSS'):
        mss_split_bin2raw(mss_binfiles[i], export_dir)


def batch_extract_pan(pan_binfiles, export_dir):
    for i in tqdm(range(len(pan_binfiles)), desc='Extract PAN'):
        pan_bin2raw(pan_binfiles[i], export_dir)


def batch_extract(input_folder, export_dir):
    file_list = os.listdir(input_folder)
    pan_list = []
    mss_list = []
    for file in file_list:
        if file.endswith('bin') and 'w8696_h4000'in file:
            pan_list.append(os.path.join(input_folder, file))
        if file.endswith('bin') and 'w2272_h1000'in file:
            mss_list.append(os.path.join(input_folder, file))
    if pan_list:
        batch_extract_pan(pan_list, export_dir)
    if mss_list:
        batch_extract_mss(mss_list, export_dir)



if __name__ == "__main__":
    input_folder = r'E:\RSDATA\亮度数据'
    output_dir = r'E:\RSDATA\亮度数据'
    batch_extract(input_folder, output_dir)