import numpy as np
import pandas as pd
import matplotlib
from PIL import Image, ImageTk
# -------------------- 图像读写辅助 --------------------
def imread(path):
    """读取单波段图像为二维numpy数组，支持tif和常见格式"""
    try:
        import tifffile
        img = tifffile.imread(path)
    except ImportError:
        img = np.array(Image.open(path))
    if img.ndim == 3:
        # 如果是3通道，转为灰度（取第一个通道）
        img = img[:,:,0]
    return img.astype(np.float64)

def imwrite(path, data):
    """保存校正图像，保持原始数据类型（尽可能）"""
    # 保存为tif，使用tifffile或PIL
    try:
        import tifffile
        tifffile.imwrite(path, data)
    except ImportError:
        Image.fromarray(data).save(path)

# -------------------- 核心函数 --------------------
def load_coefficients(coeff_file):
    df = pd.read_csv(coeff_file)
    if 'Gain' not in df.columns or 'Offset' not in df.columns:
        raise ValueError("系数文件必须包含 'Gain' 和 'Offset' 列。")
    G = df['Gain'].values.astype(np.float64)
    O = df['Offset'].values.astype(np.float64)
    return G, O

def apply_correction(image, G, O):
    return G * image + O

img_file = r'E:/RSDATA/亮度数据/CMOS_B/B3_TDI16GAIN2/亮度1/tif/B1_00000002_000000000111C1C9_w2272_h1000_pMono12.tif'
coeff_file = r'E:/RSDATA/亮度数据/CMOS_B/B3_TDI16GAIN2_coef.csv'
# 读取图像
img_raw = imread(img_file)
if img_raw.ndim != 2:
    raise RuntimeError("只支持单波段（二维）图像，当前图像维度: {}".format(img_raw.ndim))
K, N = img_raw.shape

# 读取系数
G, O = load_coefficients(coeff_file)
if len(G) != N:
    raise RuntimeError(f"系数维度 ({len(G)}) 与图像列数 ({N}) 不一致。")
G = G
O = O

# 校正
img_corr = apply_correction(img_raw, G, O)