# --*-- conding:utf-8 --*--
# @File  : plot_image_mean.py
# @Author: FH
# @Date  : 2026/9/23
# @Desc  :
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def tif_col_mean_plot_pil(tif_file_path):
    im = Image.open(tif_file_path)
    img = np.array(im)
    height, width = img.shape
    print(f"图像：高(行数)={height}, 宽(列数)={width}")

    column_mean = np.mean(img, axis=0)
    column_ids = np.arange(width)

    plt.figure(figsize=(14, 5))
    plt.scatter(column_ids, column_mean, s=6, c="darkgreen", alpha=0.7)
    plt.xlabel("Column Number")
    plt.ylabel("Column Mean Value")
    plt.title("TIF Column Mean")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(tif_file_path.replace('.tif', '_plot.png'))
    return column_ids, column_mean

if __name__ == "__main__":
    tif_path = r"F:\辐射定标20260723\Uniformity\A_CMOS\B\增益2\TDI64\亮度7\tif\B1_00000000_000000000199B62B_w2272_h1000_pMono12_c.tif"
    col_idx, col_mean = tif_col_mean_plot_pil(tif_path)
