# --*-- conding:utf-8 --*--
# @File  : merge_bands.py
# @Author: FH
# @Date  : 2026/10/12
# @Desc  :
import rasterio
import numpy as np


def merge_4band_tif(band_paths, out_path, band_names=None, dtype=None,
                    compress='LZW', nodata=None):
    """
    将 4 个单波段 TIF 合成为一个 4 波段 TIF。

    参数
    ----------
    band_paths : list[str]
        4 个单波段 TIF 的路径，顺序即输出波段顺序，例如 [b1, b2, b3, b4]。
    out_path : str
        输出 4 波段 TIF 路径。
    band_names : list[str], 可选
        波段名称，例如 ['b1', 'b2', 'b3', 'b4']。会写入元数据 description。
    dtype : str, 可选
        输出数据类型，如 'uint16'。默认沿用第一个波段的 dtype。
    compress : str, 可选
        压缩方式，默认 'LZW'。可选 'DEFLATE'、'NONE' 等。
    nodata : 数值, 可选
        输出 NoData 值。默认沿用第一个波段的 nodata。

    返回
    -------
    out_path : str
        输出文件路径。
    """
    if len(band_paths) != 4:
        raise ValueError(f"需要 4 个单波段影像，当前传入 {len(band_paths)} 个")

    # 1. 打开第一个波段，读取参考地理信息
    with rasterio.open(band_paths[0]) as src0:
        profile = src0.profile.copy()
        height = src0.height
        width = src0.width
        ref_transform = src0.transform
        ref_crs = src0.crs
        ref_dtype = src0.dtypes[0]
        ref_nodata = src0.nodata
        ref_bounds = src0.bounds

    if dtype is None:
        dtype = ref_dtype
    if nodata is None:
        nodata = ref_nodata

    # 2. 检查所有波段尺寸、地理信息是否一致
    for i, p in enumerate(band_paths):
        with rasterio.open(p) as src:
            if src.width != width or src.height != height:
                raise ValueError(
                    f"波段 {i} 尺寸不一致: {src.width}x{src.height} "
                    f"!= {width}x{height} ({p})"
                )
            if src.transform != ref_transform:
                raise ValueError(f"波段 {i} 仿射变换不一致: {p}")
            if src.crs != ref_crs:
                raise ValueError(f"波段 {i} CRS 不一致: {p}")

    # 3. 更新输出 profile
    profile.update({
        'count': 4,
        'dtype': dtype,
        'compress': compress,
        'nodata': nodata,
    })

    # 4. 写入输出
    with rasterio.open(out_path, 'w', **profile) as dst:
        for i, p in enumerate(band_paths, start=1):
            with rasterio.open(p) as src:
                data = src.read(1)
            # 若 dtype 与源不同，做类型转换
            if data.dtype != np.dtype(dtype):
                data = data.astype(dtype)
            dst.write(data, i)
            if band_names is not None and i - 1 < len(band_names):
                dst.set_band_description(i, band_names[i - 1])

    return out_path


# ================== 使用示例 ==================
if __name__ == "__main__":
    band_paths = [
        r"F:\辐射定标20260723\Uniformity\A_CMOS\B\增益3\TDI4\亮度1\tif/B1_00000000_0000000000582A50_w2272_h1000_pMono12.tif",
        r"F:\辐射定标20260723\Uniformity\A_CMOS\B\增益3\TDI4\亮度1\tif/B2_00000000_0000000000582A50_w2272_h1000_pMono12.tif",
        r"F:\辐射定标20260723\Uniformity\A_CMOS\B\增益3\TDI4\亮度1\tif/B3_00000000_0000000000582A50_w2272_h1000_pMono12.tif",
        r"F:\辐射定标20260723\Uniformity\A_CMOS\B\增益3\TDI4\亮度1\tif/B4_00000000_0000000000582A50_w2272_h1000_pMono12.tif",
    ]
    merge_4band_tif(
        band_paths,
        r"F:\辐射定标20260723\Uniformity\A_CMOS\B\增益3\TDI4\亮度1\tif/PMS1_MSS_00000000_0000000000582A50_w2272_h1000_pMono12.tif",
        band_names=['b1', 'b2', 'b3', 'b4'],
        dtype='uint16',
        # compress='LZW',
    )