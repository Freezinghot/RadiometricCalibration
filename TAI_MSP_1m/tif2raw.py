#!/usr/bin/env python3
"""
将 TIFF 图像转换为 RAW 格式（原始像素数据，按行顺序存储）。
输出文件不包含任何头信息，仅包含像素数据。
"""
import os
import tifffile
from tqdm import tqdm

def tif2raw_batch(input_dir, output_dir):
    files = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    valid_file = []
    for file in files:
        if label in file:
            valid_file.append(file)
    for file in tqdm(valid_file):
        filename = os.path.join(input_dir, file)
        out_file = os.path.join(output_dir, file.replace(".tif", ".raw"))
        # 读取 TIFF 图像（返回 numpy 数组）
        img = tifffile.imread(filename)

        # 将数组以二进制格式写入文件（按内存顺序，通常为行主序）
        img.tofile(out_file)


if __name__ == "__main__":
    input_dir = r'F:\辐射定标20260723\均匀性\B_CMOS\暗场\B\增益2\TDI64\tif'
    output_dir = r'F:\辐射定标20260723\均匀性\B_CMOS\暗场\B\增益2\TDI64\raw'
    label = 'w2272_h1000_pMono12.tif'
    tif2raw_batch(input_dir, output_dir)