# --*-- conding:utf-8 --*--
# @File  : radiance_equivalent.py
# @Author: FH
# @Date  : 2026/8/24
# @Desc  : 辐亮度等效波段辐照度
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simpson, trapezoid
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, medfilt

def preprocess_spectral_response(response_file, response_modify, step=1.0,
                                 median_window=5, sg_window=15, sg_polyorder=3,
                                 normalize=True):
    """
    预处理光谱响应度数据：统一采样间隔为 step nm，去除异常值，平滑曲线。

    参数
    ----
    wavelength : array_like
        原始波长数据（单位通常为 nm），形状 (N,)。
    response : array_like
        原始响应度数据，形状与 wavelength 相同。
    step : float, 可选
        目标采样间隔，默认 1.0 nm。
    median_window : int, 可选
        中值滤波窗口大小（用于去除尖峰异常），必须为奇数，默认 5。
    sg_window : int, 可选
        Savitzky-Golay 滤波窗口长度，必须为奇数，且不大于数据点数，默认 15。
    sg_polyorder : int, 可选
        Savitzky-Golay 滤波多项式阶数，默认 3。
    normalize : bool, 可选
        是否将处理后的响应归一化到最大值 1，默认 True。

    返回
    ----
    wavelength_out : np.ndarray
        均匀采样后的波长数组（间隔为 step）。
    response_out : np.ndarray
        预处理后的响应度数组，非负，已平滑。
    """
    # 假设原始数据存储在 CSV 文件中
    df = pd.read_csv(response_file)
    wavelength = df['wavelength'].values
    response = df['response'].values
    # 转换为 numpy 数组并展平
    wavelength = np.asarray(wavelength, dtype=float).ravel()
    response = np.asarray(response, dtype=float).ravel()

    # 检查长度一致
    if wavelength.shape[0] != response.shape[0]:
        raise ValueError("wavelength 和 response 长度不一致。")

    # 1. 按波长排序
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    response = response[sort_idx]

    # 2. 去除无效值（NaN, Inf）
    valid = np.isfinite(wavelength) & np.isfinite(response)
    wavelength = wavelength[valid]
    response = response[valid]

    if len(wavelength) < 3:
        raise ValueError("有效数据点太少，无法进行插值和平滑。")

    # 3. 负值置零（响应度物理上不应为负）
    response = np.where(response < 0, 0.0, response)

    # 4. 插值到均匀网格
    # 确定插值范围：取原始波长最小值和最大值
    wl_min = np.floor(wavelength.min())
    wl_max = np.ceil(wavelength.max())
    wavelength_grid = np.arange(wl_min, wl_max + step, step)

    # 线性插值（超出范围填充 0）
    interp_func = interp1d(wavelength, response, kind='linear',
                           fill_value=0.0, bounds_error=False)
    response_grid = interp_func(wavelength_grid)

    # 5. 中值滤波去除异常尖峰（此时数据均匀，中值滤波有效）
    # 确保窗口为奇数且不大于数据点数
    if median_window % 2 == 0:
        median_window += 1
    if median_window > len(response_grid):
        median_window = len(response_grid) if len(response_grid) % 2 == 1 else len(response_grid) - 1
    if median_window >= 3:
        response_grid = medfilt(response_grid, kernel_size=median_window)

    # 6. Savitzky-Golay 平滑滤波
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window > len(response_grid):
        sg_window = len(response_grid) if len(response_grid) % 2 == 1 else len(response_grid) - 1
    if sg_polyorder >= sg_window:
        sg_polyorder = sg_window - 1
    if sg_window >= 3 and sg_polyorder >= 1:
        response_grid = savgol_filter(response_grid, sg_window, sg_polyorder)

    # 7. 再次确保非负
    response_grid = np.where(response_grid < 0, 0.0, response_grid)

    # 8. 可选归一化
    if normalize:
        max_val = np.max(response_grid)
        if max_val > 0:
            response_grid = response_grid / max_val

    # 保存结果
    return pd.DataFrame({'wavelength': wavelength_grid, 'response': response_grid}).to_csv(
        response_modify, index=False
    )

def equivalent_radiance(radiance, response):
    # 读取数据
    df_L = pd.read_csv(radiance)  # 列: wavelength, radiance
    df_R = pd.read_csv(response)  # 列: wavelength, response

    lambda_L = df_L['wavelength'].values
    L = df_L['radiance'].values
    lambda_R = df_R['wavelength'].values
    R = df_R['response'].values

    # 1. 确定公共波长网格：取两者波长范围的交集，并适当外扩以包含带外响应
    lambda_min = max(lambda_L.min(), lambda_R.min())
    lambda_max = min(lambda_L.max(), lambda_R.max())
    # 可生成等间隔网格（步长取两表最小间隔或 1 nm）
    step = min(np.min(np.diff(lambda_L)), np.min(np.diff(lambda_R)), 1.0)
    lambda_grid = np.arange(lambda_min, lambda_max + step, step)

    # 2. 插值到公共网格
    interp_L = interp1d(lambda_L, L, kind='linear', fill_value=0.0, bounds_error=False)
    interp_R = interp1d(lambda_R, R, kind='linear', fill_value=0.0, bounds_error=False)

    L_grid = interp_L(lambda_grid)
    R_grid = interp_R(lambda_grid)

    # 3. 计算加权积分
    # 分子：∫ L(λ) R(λ) dλ
    integrand = L_grid * R_grid

    # 梯形法（适用于非等间隔，这里等间隔也可用）
    integral_weighted = np.trapz(integrand, lambda_grid)

    # 分母：∫ R(λ) dλ
    integral_response = np.trapz(R_grid, lambda_grid)

    # 波段等效辐亮度
    Leff = integral_weighted / integral_response

    print(f"波段等效辐亮度 L_eff = {Leff:.6f} W·m⁻²·sr⁻¹")
    print(f"有效带宽 Δλ_eff = {integral_response / np.max(R_grid):.2f} nm")

if __name__ == "__main__":
    radiance_file = r'F:\辐射定标20260723\辅助数据\radiance.CSV'
    response_file = r'F:\辐射定标20260723\辅助数据\radiance_response_A_P_smooth.csv'
    response_modify = response_file.replace('.csv', '_smooth.csv')
    # preprocess_spectral_response(response_file, response_modify)
    equivalent_radiance(radiance_file, response_file)
