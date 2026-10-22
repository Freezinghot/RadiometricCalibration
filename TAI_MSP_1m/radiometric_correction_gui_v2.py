# --*-- conding:utf-8 --*--
# @File  : radiometric_correction_gui_v2.py
# @Author: FH
# @Date  : 2026/10/9
# @Desc  :
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

# -------------------- 图像读写辅助 --------------------
def imread(path):
    try:
        import tifffile
        img = tifffile.imread(path)
    except ImportError:
        img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float64)

def imwrite(path, data):
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

# ===== 新增：条纹系数（逐列） =====
def compute_streaking_metric(col_mean):
    """
    条纹系数定义：|μ_i - (μ_{i-1}+μ_{i+1})/2| / ((μ_{i-1}+μ_{i+1})/2) * 100%
    边界列退化为与相邻单列比较。
    返回与 col_mean 等长的一维数组。
    """
    n = len(col_mean)
    s = np.zeros(n, dtype=np.float64)
    if n < 2:
        return s
    if n >= 3:
        ref = (col_mean[:-2] + col_mean[2:]) / 2.0
        valid = ref != 0
        tmp = np.zeros(len(ref), dtype=np.float64)
        tmp[valid] = np.abs(col_mean[1:-1][valid] - ref[valid]) / ref[valid] * 100.0
        s[1:-1] = tmp
    # if col_mean[1] != 0:
        # s[0] = abs(col_mean[0] - col_mean[1]) / col_mean[1] * 100.0
    s[0] = 0
    # if col_mean[-2] != 0:
        # s[-1] = abs(col_mean[-1] - col_mean[-2]) / col_mean[-2] * 100.0
    s[-1] = 0
    return s

# ===== 新增：列均值均方根 RMS（归一化，单值） =====
def compute_column_rms(col_mean):
    m = np.mean(col_mean)
    n = len(col_mean)
    if m == 0 or n < 2:
        return 0.0
    return np.sqrt(np.sum((col_mean - m) ** 2) / (n - 1)) / m * 100.0

def evaluate_uniformity(img_raw, img_corr):
    # 1. 原有按列统计
    col_mean_raw = np.mean(img_raw, axis=0)
    col_mean_corr = np.mean(img_corr, axis=0)
    col_std_raw = np.std(img_raw, axis=0, ddof=1)
    col_std_corr = np.std(img_corr, axis=0, ddof=1)

    global_mean_raw = np.mean(img_raw)
    global_mean_corr = np.mean(img_corr)
    global_std_raw = np.std(img_raw, ddof=1)
    global_std_corr = np.std(img_corr, ddof=1)

    streaking_raw = np.std(col_mean_raw, ddof=1)
    streaking_corr = np.std(col_mean_corr, ddof=1)

    cv_raw = global_std_raw / global_mean_raw if global_mean_raw != 0 else np.nan
    cv_corr = global_std_corr / global_mean_corr if global_mean_corr != 0 else np.nan

    # 2. 逐行 CV
    row_mean_raw = np.mean(img_raw, axis=1)
    row_mean_corr = np.mean(img_corr, axis=1)
    row_std_raw = np.std(img_raw, axis=1, ddof=1)
    row_std_corr = np.std(img_corr, axis=1, ddof=1)

    mask_raw = row_mean_raw > 1e-6
    mask_corr = row_mean_corr > 1e-6
    row_cv_raw = np.full_like(row_mean_raw, np.nan)
    row_cv_corr = np.full_like(row_mean_corr, np.nan)
    row_cv_raw[mask_raw] = row_std_raw[mask_raw] / row_mean_raw[mask_raw]
    row_cv_corr[mask_corr] = row_std_corr[mask_corr] / row_mean_corr[mask_corr]

    # ===== 新增：条纹系数（逐列） =====
    streaking_metric_raw = compute_streaking_metric(col_mean_raw)
    streaking_metric_corr = compute_streaking_metric(col_mean_corr)

    # ===== 新增：列均值均方根 RMS =====
    column_rms_raw = compute_column_rms(col_mean_raw)
    column_rms_corr = compute_column_rms(col_mean_corr)

    metrics = {
        'global_mean_raw': global_mean_raw,
        'global_mean_corr': global_mean_corr,
        'global_std_raw': global_std_raw,
        'global_std_corr': global_std_corr,
        'global_heterogeneity_corr': global_std_corr / global_mean_corr,
        'streaking_raw': streaking_raw,
        'streaking_corr': streaking_corr,
        'cv_raw': cv_raw,
        'cv_corr': cv_corr,
        'col_mean_raw': col_mean_raw,
        'col_mean_corr': col_mean_corr,
        'col_std_raw': col_std_raw,
        'col_std_corr': col_std_corr,
        'row_cv_raw': row_cv_raw,
        'row_cv_corr': row_cv_corr,
        # ===== 新增字段 =====
        'streaking_metric_raw': streaking_metric_raw,
        'streaking_metric_corr': streaking_metric_corr,
        'column_rms_raw': column_rms_raw,
        'column_rms_corr': column_rms_corr,
    }
    return metrics

# -------------------- GUI 应用类 --------------------
class VerifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("辐射校正系数验证工具")
        self.root.geometry("1000x900")

        self.image_path = tk.StringVar()
        self.coeff_path = tk.StringVar()
        self.img_raw = None
        self.img_corr = None
        self.metrics = None
        self.G = None
        self.O = None

        self.create_widgets()

    def create_widgets(self):
        frame_files = ttk.LabelFrame(self.root, text="输入文件", padding=10)
        frame_files.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame_files, text="原始图像:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(frame_files, textvariable=self.image_path, width=70).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(frame_files, text="浏览...", command=self.select_image).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(frame_files, text="系数文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(frame_files, textvariable=self.coeff_path, width=70).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(frame_files, text="浏览...", command=self.select_coeff).grid(row=1, column=2, padx=5, pady=2)

        frame_btn = ttk.Frame(self.root)
        frame_btn.pack(fill=tk.X, padx=10, pady=5)

        self.btn_run = ttk.Button(frame_btn, text="执行验证", command=self.run_verification)
        self.btn_run.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(frame_btn, text="保存校正图像", state=tk.DISABLED, command=self.save_corrected)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame_btn, text="退出", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        frame_text = ttk.LabelFrame(main_panel, text="均匀性指标", padding=5)
        main_panel.add(frame_text, weight=1)

        self.text_metrics = tk.Text(frame_text, wrap=tk.NONE, state=tk.DISABLED, height=26, width=48)
        scroll_y = ttk.Scrollbar(frame_text, command=self.text_metrics.yview)
        self.text_metrics.configure(yscrollcommand=scroll_y.set)
        self.text_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        frame_plot = ttk.LabelFrame(main_panel, text="分析图", padding=5)
        main_panel.add(frame_plot, weight=3)

        # ===== 修改：子图由 4 行改为 5 行 =====
        self.fig, self.axes = plt.subplots(5, 1, figsize=(7, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.clear_plot()

    def select_image(self):
        filename = filedialog.askopenfilename(
            title="选择原始图像",
            filetypes=[("图像文件", "*.tif *.tiff *.png *.bmp *.jpg"), ("所有文件", "*.*")]
        )
        if filename:
            self.image_path.set(filename)

    def select_coeff(self):
        filename = filedialog.askopenfilename(
            title="选择系数CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if filename:
            self.coeff_path.set(filename)

    def run_verification(self):
        img_file = self.image_path.get()
        coeff_file = self.coeff_path.get()
        if not img_file:
            messagebox.showerror("错误", "请选择原始图像文件。")
            return
        if not coeff_file:
            messagebox.showerror("错误", "请选择系数CSV文件。")
            return

        try:
            self.status_var.set("读取图像和系数中...")
            self.root.update_idletasks()

            self.img_raw = imread(img_file)
            if self.img_raw.ndim != 2:
                raise RuntimeError("只支持单波段（二维）图像，当前维度: {}".format(self.img_raw.ndim))
            K, N = self.img_raw.shape

            G, O = load_coefficients(coeff_file)
            if len(G) != N:
                raise RuntimeError(f"系数维度 ({len(G)}) 与图像列数 ({N}) 不一致。")
            self.G = G
            self.O = O

            self.img_corr = apply_correction(self.img_raw, G, O)
            self.metrics = evaluate_uniformity(self.img_raw, self.img_corr)

            self.update_metrics_display()
            self.update_plot()

            self.btn_save.config(state=tk.NORMAL)
            self.status_var.set("验证完成")
            messagebox.showinfo("完成", "系数验证完成，可查看指标和图表。")

        except Exception as e:
            messagebox.showerror("执行失败", str(e))
            self.status_var.set("失败")

    def update_metrics_display(self):
        m = self.metrics

        rcv_raw = m['row_cv_raw']
        rcv_corr = m['row_cv_corr']
        mean_rcv_raw = np.nanmean(rcv_raw)
        mean_rcv_corr = np.nanmean(rcv_corr)
        median_rcv_raw = np.nanmedian(rcv_raw)
        median_rcv_corr = np.nanmedian(rcv_corr)
        std_rcv_raw = np.nanstd(rcv_raw)
        std_rcv_corr = np.nanstd(rcv_corr)

        # ===== 新增：条纹系数统计量 =====
        sm_raw = m['streaking_metric_raw']
        sm_corr = m['streaking_metric_corr']
        mean_sm_raw = np.mean(sm_raw)
        mean_sm_corr = np.mean(sm_corr)
        max_sm_raw = np.max(sm_raw)
        max_sm_corr = np.max(sm_corr)

        text = (
            f"原始图像全局均值: {m['global_mean_raw']:.4f}\n"
            f"校正图像全局均值: {m['global_mean_corr']:.4f}\n\n"
            f"原始图像全局标准差: {m['global_std_raw']:.4f}\n"
            f"校正图像全局标准差: {m['global_std_corr']:.4f}\n\n"
            # 修正：原来误用了 global_mean_corr，改用 heterogeneity
            f"校正图像非均匀性: {m['global_heterogeneity_corr']:.6f}\n\n"
            f"原始图像列均值标准差 (条带强度): {m['streaking_raw']:.4f}\n"
            f"校正图像列均值标准差 (条带强度): {m['streaking_corr']:.4f}\n"
            f"条带强度降低百分比: {self.get_improvement():.2f}%\n\n"
            f"原始图像全局变异系数 (CV): {m['cv_raw']:.6f}\n"
            f"校正图像全局变异系数 (CV): {m['cv_corr']:.6f}\n\n"
            # ===== 新增：列均值 RMS 数值 =====
            f"--- 列均值均方根 (RMS) ---\n"
            f"原始列均值RMS: {m['column_rms_raw']:.4f} %\n"
            f"校正列均值RMS: {m['column_rms_corr']:.4f} %\n"
            f"RMS 降低百分比: {self.get_rms_improvement():.2f}%\n\n"
            # ===== 新增：条纹系数辅助信息 =====
            f"--- 条纹系数 (Streaking Metric) ---\n"
            f"原始 - 均值: {mean_sm_raw:.4f}%  最大: {max_sm_raw:.4f}%\n"
            f"校正 - 均值: {mean_sm_corr:.4f}%  最大: {max_sm_corr:.4f}%\n"
            f"(验收参考线: 0.25%)\n\n"
            f"--- 逐行 CV (行标准差/行均值) ---\n"
            f"原始 - 均值: {mean_rcv_raw:.6f}  中位数: {median_rcv_raw:.6f}  标准差: {std_rcv_raw:.6f}\n"
            f"校正 - 均值: {mean_rcv_corr:.6f}  中位数: {median_rcv_corr:.6f}  标准差: {std_rcv_corr:.6f}\n"
        )
        self.text_metrics.config(state=tk.NORMAL)
        self.text_metrics.delete(1.0, tk.END)
        self.text_metrics.insert(tk.END, text)
        self.text_metrics.config(state=tk.DISABLED)

    def get_improvement(self):
        raw = self.metrics['streaking_raw']
        corr = self.metrics['streaking_corr']
        if raw == 0:
            return 0.0
        return (1 - corr / raw) * 100

    # ===== 新增：RMS 降低百分比 =====
    def get_rms_improvement(self):
        raw = self.metrics['column_rms_raw']
        corr = self.metrics['column_rms_corr']
        if raw == 0:
            return 0.0
        return (1 - corr / raw) * 100

    def update_plot(self):
        for ax in self.axes:
            ax.clear()

        m = self.metrics
        N = len(m['col_mean_raw'])
        K = len(m['row_cv_raw'])
        x_col = np.arange(N)
        x_row = np.arange(K)

        # 1. 列均值对比
        self.axes[0].plot(x_col, m['col_mean_raw'], 'r-', alpha=0.7, linewidth=0.8, label='before')
        self.axes[0].plot(x_col, m['col_mean_corr'], 'b-', alpha=0.7, linewidth=0.8, label='after')
        self.axes[0].set_title('Column mean')
        self.axes[0].set_ylabel('DN mean')
        self.axes[0].set_xlabel('col')
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)

        # 2. 列标准差对比
        self.axes[1].plot(x_col, m['col_std_raw'], 'r-', alpha=0.7, linewidth=0.8, label='before')
        self.axes[1].plot(x_col, m['col_std_corr'], 'b-', alpha=0.7, linewidth=0.8, label='after')
        self.axes[1].set_title('Column Standard Deviations')
        self.axes[1].set_ylabel('DN Standard Deviations')
        self.axes[1].set_xlabel('col')
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)

        # 3. 全局直方图
        self.axes[2].hist(self.img_raw.ravel(), bins=200, alpha=0.5, color='red', label='before', density=True)
        self.axes[2].hist(self.img_corr.ravel(), bins=200, alpha=0.5, color='blue', label='after', density=True)
        self.axes[2].set_title('Global Histogram')
        self.axes[2].set_xlabel('DN')
        self.axes[2].set_ylabel('Percentage')
        self.axes[2].legend()
        self.axes[2].grid(True, alpha=0.3)

        # 4. 逐行 CV
        self.axes[3].plot(x_row, m['row_cv_raw'], 'r-', alpha=0.7, linewidth=0.8, label='before')
        self.axes[3].plot(x_row, m['row_cv_corr'], 'b-', alpha=0.7, linewidth=0.8, label='after')
        self.axes[3].set_title('Row Coefficient of Variation')
        self.axes[3].set_xlabel('row')
        self.axes[3].set_ylabel('CV')
        self.axes[3].legend()
        self.axes[3].grid(True, alpha=0.3)

        # ===== 新增：5. 条纹系数曲线 =====
        self.axes[4].plot(x_col, m['streaking_metric_raw'], 'r-', alpha=0.6, linewidth=0.8, label='before')
        self.axes[4].plot(x_col, m['streaking_metric_corr'], 'b-', alpha=0.8, linewidth=0.8, label='after')
        self.axes[4].axhline(0.25, color='green', linestyle='--', linewidth=1, label='0.25% threshold')
        self.axes[4].set_title('Streaking Metric (Column-wise)')
        self.axes[4].set_xlabel('col')
        self.axes[4].set_ylabel('Streaking (%)')
        self.axes[4].set_ylim(0, 1)  # 新增：纵轴上限固定为 1
        self.axes[4].legend()
        self.axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        self.canvas.draw()

    def clear_plot(self):
        for ax in self.axes:
            ax.clear()
            ax.text(0.5, 0.5, '请加载数据并执行验证', ha='center', va='center', transform=ax.transAxes)
        self.canvas.draw()

    def save_corrected(self):
        if self.img_corr is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIF图像", "*.tif"), ("所有文件", "*.*")],
            title="保存校正图像"
        )
        if not file_path:
            return
        try:
            raw_dtype = imread(self.image_path.get()).dtype
            if np.issubdtype(raw_dtype, np.integer):
                info = np.iinfo(raw_dtype)
                data = np.clip(np.round(self.img_corr), info.min, info.max).astype(raw_dtype)
            else:
                data = self.img_corr.astype(np.float32)
            imwrite(file_path, data)
            self.status_var.set(f"校正图像已保存至: {file_path}")
            messagebox.showinfo("保存成功", f"文件已保存至:\n{file_path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

# -------------------- 主入口 --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = VerifierApp(root)
    root.mainloop()