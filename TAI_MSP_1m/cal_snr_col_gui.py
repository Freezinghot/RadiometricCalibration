# --*-- conding:utf-8 --*--
# @File  : cal_snr_col_gui.py
# @Author: FH
# @Date  : 2026/8/19
# @Desc  :
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

# 绘图相关
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================== 图像读取 ====================

def read_tif(path):
    """读取 TIFF 图像，转为 16 位灰度矩阵，范围 0-4095"""
    img = Image.open(path)
    if img.mode == 'I;16':
        arr = np.array(img, dtype=np.uint16)
    elif img.mode == 'I':
        arr = np.array(img, dtype=np.int32)
        arr = np.clip(arr, 0, 4095).astype(np.uint16)
    else:
        img = img.convert('I')
        arr = np.array(img, dtype=np.int32)
        arr = np.clip(arr, 0, 4095).astype(np.uint16)
    return np.clip(arr, 0, 4095)


# ==================== 坏列检测与修复 ====================

def detect_bad_columns(img, valid_mask=None, k=3, max_iter=3):
    """先裁剪有效列，再进行逐列均值3σ检测"""
    if valid_mask is None:
        valid_mask = np.ones_like(img, dtype=bool)

    col_valid = np.any(valid_mask, axis=0)
    valid_indices = np.where(col_valid)[0]
    if len(valid_indices) == 0:
        return np.zeros(img.shape[1], dtype=bool)

    left = valid_indices[0]
    right = valid_indices[-1] + 1

    sub_img = img[:, left:right]
    sub_valid = valid_mask[:, left:right]

    h, w_sub = sub_img.shape
    data = sub_img.astype(np.float64)

    col_means = np.full(w_sub, np.nan)
    for j in range(w_sub):
        col_data = data[sub_valid[:, j], j]
        if len(col_data) > 0:
            col_means[j] = np.mean(col_data)

    bad_cols_sub = np.zeros(w_sub, dtype=bool)
    for _ in range(max_iter):
        current = ~bad_cols_sub & ~np.isnan(col_means)
        if np.sum(current) < 2:
            break
        mu = np.mean(col_means[current])
        sigma = np.std(col_means[current])
        if sigma == 0:
            break
        new_bad = np.zeros(w_sub, dtype=bool)
        new_bad[current] = np.abs(col_means[current] - mu) > k * sigma
        if np.array_equal(new_bad, bad_cols_sub):
            break
        bad_cols_sub = new_bad

    bad_cols = np.zeros(img.shape[1], dtype=bool)
    bad_cols[left:right] = bad_cols_sub
    return bad_cols


def correct_bad_columns(img, bad_cols, valid_mask=None):
    """整列替换坏列：用左右最近的有效非坏列对应像素的平均值替换"""
    if valid_mask is None:
        valid_mask = np.ones_like(img, dtype=bool)

    corrected = img.copy().astype(np.float64)
    h, w = img.shape

    col_valid = np.any(valid_mask, axis=0)
    good_cols = col_valid & ~bad_cols
    bad_indices = np.where(col_valid & bad_cols)[0]

    for col in bad_indices:
        left_col = None
        for c in range(col - 1, -1, -1):
            if good_cols[c]:
                left_col = c
                break
        right_col = None
        for c in range(col + 1, w):
            if good_cols[c]:
                right_col = c
                break

        if left_col is not None and right_col is not None:
            corrected[:, col] = (corrected[:, left_col] + corrected[:, right_col]) / 2.0
        elif left_col is not None:
            corrected[:, col] = corrected[:, left_col]
        elif right_col is not None:
            corrected[:, col] = corrected[:, right_col]

    return np.clip(corrected, 0, 4095).astype(np.uint16)


# ==================== 逐列 SNR 计算 ====================

def compute_column_snr_avg(img, valid_mask, col_valid, min_valid_pixels=10, eps=1e-6):
    """
    逐列计算列内 SNR，返回平均 SNR 和每列 SNR 数组。
    参数：
        img: 图像数组
        valid_mask: 像素级有效掩码
        col_valid: 列有效掩码（bool 数组，长度等于列数）
        min_valid_pixels: 列内有效像素数小于该值的列不参与计算
        eps: 防止除零的小量
    返回：
        avg_snr_db: 有效列 SNR 的平均值
        snr_per_col: 长度等于总列数的数组，无效列或有效像素不足的列为 NaN
    """
    h, w = img.shape
    data = img.astype(np.float64)
    snr_values = np.full(w, np.nan)
    valid_cols_for_snr = []

    for col in range(w):
        if not col_valid[col]:
            continue
        # 该列有效像素
        col_mask = valid_mask[:, col]
        col_data = data[col_mask, col]
        if len(col_data) < min_valid_pixels:
            continue
        mu = np.mean(col_data)
        sigma = np.std(col_data)
        if sigma < eps:
            snr = 20 * np.log10(mu / eps)  # 噪声极小，给一个大值
        else:
            snr = 20 * np.log10(mu / sigma)
        snr_values[col] = snr
        valid_cols_for_snr.append(col)

    if len(valid_cols_for_snr) == 0:
        avg_snr_db = np.nan
    else:
        avg_snr_db = np.nanmean(snr_values[valid_cols_for_snr])
    return avg_snr_db, snr_values


# ==================== GUI 主程序 ====================

class SNRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像信噪比计算工具（逐列SNR）")
        self.root.geometry("900x850")

        self.image = None
        self.valid_mask = None
        self.bad_cols = None
        self.corrected = None
        self.file_path = None

        self.create_widgets()

    def create_widgets(self):
        # 文件选择
        file_frame = ttk.LabelFrame(self.root, text="输入 TIFF 文件", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(file_frame, text="选择文件", command=self.select_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=5)

        # 有效区域设置
        valid_frame = ttk.LabelFrame(self.root, text="有效区域设置（统计时剔除左右无效列）", padding=10)
        valid_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(valid_frame, text="左侧无效列数:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.left_invalid_var = tk.StringVar(value='0')
        ttk.Entry(valid_frame, textvariable=self.left_invalid_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(valid_frame, text="右侧无效列数:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.right_invalid_var = tk.StringVar(value='0')
        ttk.Entry(valid_frame, textvariable=self.right_invalid_var, width=8).grid(row=0, column=3, padx=5)

        # 操作按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="加载图像", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="检测坏列", command=self.detect_bad_cols).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="修复坏列", command=self.correct_bad_cols).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="计算列SNR", command=self.calculate_column_snr).pack(side=tk.LEFT, padx=5)

        # 保存功能
        save_frame = ttk.Frame(self.root)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(save_frame, text="保存当前图像", command=self.save_image).pack(side=tk.LEFT, padx=5)
        self.save_valid_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(save_frame, text="仅保存有效列", variable=self.save_valid_only_var).pack(side=tk.LEFT, padx=5)

        # 坏列检测参数
        param_frame = ttk.LabelFrame(self.root, text="坏列检测参数", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(param_frame, text="阈值倍数 k:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.k_var = tk.StringVar(value='3')
        ttk.Entry(param_frame, textvariable=self.k_var, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="(逐列均值3σ)").grid(row=0, column=2, sticky=tk.W, padx=5)

        # 图像显示区域
        self.image_panel = ttk.Label(self.root, text="图像预览区域", anchor=tk.CENTER)
        self.image_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 结果输出
        result_frame = ttk.LabelFrame(self.root, text="结果", padding=10)
        result_frame.pack(fill=tk.X, padx=10, pady=5)
        self.result_text = tk.Text(result_frame, height=5, width=90)
        self.result_text.pack(fill=tk.X)

        # 绘图区域
        plot_frame = ttk.LabelFrame(self.root, text="逐列信噪比", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF文件", "*.tif *.tiff"), ("所有文件", "*.*")])
        if path:
            self.file_path = path
            self.file_label.config(text=path)

    def load_image(self):
        if not self.file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return
        try:
            self.image = read_tif(self.file_path)
            self.bad_cols = None
            self.corrected = None
            self.setup_valid_mask()
            self.display_image(self.image)

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"图像加载成功，尺寸: {self.image.shape[1]} x {self.image.shape[0]}\n")
            self.result_text.insert(tk.END, f"灰度范围: {self.image.min()} - {self.image.max()}\n")
            if self.valid_mask is not None:
                valid_pixels = np.sum(self.valid_mask)
                self.result_text.insert(tk.END, f"有效像素数（剔除无效列后）: {valid_pixels}\n")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def setup_valid_mask(self):
        if self.image is None:
            self.valid_mask = None
            return
        h, w = self.image.shape
        valid = np.ones((h, w), dtype=bool)
        left = int(self.left_invalid_var.get() or 0)
        right = int(self.right_invalid_var.get() or 0)
        if left > 0:
            left = min(left, w)
            valid[:, :left] = False
        if right > 0:
            right = min(right, w)
            valid[:, w - right:] = False
        self.valid_mask = valid

    def display_image(self, img):
        disp = ((img.astype(np.float64) / 4095.0) * 255).astype(np.uint8)
        if self.valid_mask is not None:
            disp_rgb = np.stack([disp, disp, disp], axis=2)
            invalid = ~self.valid_mask
            disp_rgb[invalid, 2] = 255   # 无效列标蓝
            disp_rgb[invalid, 0] = 0
            disp_rgb[invalid, 1] = 0
            disp = disp_rgb
        pil_img = Image.fromarray(disp, mode='RGB' if disp.ndim == 3 else 'L')
        pil_img.thumbnail((500, 400))
        imgtk = ImageTk.PhotoImage(pil_img)
        self.image_panel.config(image=imgtk)
        self.image_panel.image = imgtk

    def detect_bad_cols(self):
        if self.image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        try:
            k = float(self.k_var.get())
            self.setup_valid_mask()
            self.bad_cols = detect_bad_columns(self.image, valid_mask=self.valid_mask, k=k)
            num_bad = np.sum(self.bad_cols)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"坏列检测完成（逐列均值3σ，k={k}）\n")
            self.result_text.insert(tk.END, f"检测到坏列数量: {num_bad}\n")
            if num_bad > 0:
                bad_idx = np.where(self.bad_cols)[0]
                self.result_text.insert(tk.END, f"坏列序号: {bad_idx.tolist()}\n")
            self.display_with_bad_cols()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def display_with_bad_cols(self):
        if self.image is None or self.bad_cols is None:
            return
        disp = ((self.image.astype(np.float64) / 4095.0) * 255).astype(np.uint8)
        rgb = np.stack([disp, disp, disp], axis=2)
        bad_col_mask = np.tile(self.bad_cols, (self.image.shape[0], 1))
        rgb[bad_col_mask, 0] = 255
        rgb[bad_col_mask, 1] = 0
        rgb[bad_col_mask, 2] = 0
        if self.valid_mask is not None:
            invalid = ~self.valid_mask
            rgb[invalid, 0] = 0
            rgb[invalid, 1] = 0
            rgb[invalid, 2] = 255
        pil_img = Image.fromarray(rgb, mode='RGB')
        pil_img.thumbnail((500, 400))
        imgtk = ImageTk.PhotoImage(pil_img)
        self.image_panel.config(image=imgtk)
        self.image_panel.image = imgtk

    def correct_bad_cols(self):
        if self.image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        if self.bad_cols is None:
            messagebox.showwarning("警告", "请先检测坏列")
            return
        try:
            self.setup_valid_mask()
            self.corrected = correct_bad_columns(self.image, self.bad_cols, valid_mask=self.valid_mask)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "坏列修复完成：坏列整列已用相邻有效列替换\n")
            self.display_image(self.corrected)
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def calculate_column_snr(self):
        if self.image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return

        # 选择计算图像：优先使用修复后的图像
        if self.corrected is not None:
            img_for_snr = self.corrected
        else:
            img_for_snr = self.image

        # 构建列有效掩码
        self.setup_valid_mask()
        col_valid = np.any(self.valid_mask, axis=0)

        # 若存在未修复的坏列，则排除它们
        if self.corrected is None and self.bad_cols is not None:
            col_valid = col_valid & ~self.bad_cols

        try:
            avg_snr, snr_per_col = compute_column_snr_avg(
                img_for_snr, self.valid_mask, col_valid,
                min_valid_pixels=10, eps=1e-6
            )
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            if np.isnan(avg_snr):
                self.result_text.insert(tk.END, "没有足够的列进行 SNR 计算\n")
            else:
                self.result_text.insert(tk.END, f"逐列 SNR 平均值: {avg_snr:.2f} dB\n")
                valid_cols = np.where(~np.isnan(snr_per_col))[0]
                self.result_text.insert(tk.END, f"参与计算的列数: {len(valid_cols)}\n")
                self.result_text.insert(tk.END, f"SNR 范围: {np.nanmin(snr_per_col):.2f} ~ {np.nanmax(snr_per_col):.2f} dB\n")

            # 绘图
            self.ax.clear()
            cols = np.arange(len(snr_per_col))
            # 只绘制有效 SNR 的点
            valid_mask_plot = ~np.isnan(snr_per_col)
            self.ax.plot(cols[valid_mask_plot], snr_per_col[valid_mask_plot],
                         marker='o', linestyle='-', markersize=2, linewidth=0.8)
            self.ax.set_xlabel('列号')
            self.ax.set_ylabel('SNR (dB)')
            self.ax.set_title('逐列信噪比')
            self.ax.grid(True, linestyle='--', alpha=0.6)
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def save_image(self):
        if self.image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return

        img_to_save = self.corrected if self.corrected is not None else self.image

        if self.save_valid_only_var.get():
            self.setup_valid_mask()
            if self.valid_mask is None:
                messagebox.showerror("错误", "有效掩码未生成")
                return
            valid_cols = np.any(self.valid_mask, axis=0)
            if not np.any(valid_cols):
                messagebox.showwarning("警告", "没有有效列可保存")
                return
            col_indices = np.where(valid_cols)[0]
            left = col_indices[0]
            right = col_indices[-1] + 1
            img_to_save = img_to_save[:, left:right]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF文件", "*.tif"), ("所有文件", "*.*")],
            title="保存图像"
        )
        if not file_path:
            return

        try:
            arr = np.clip(img_to_save, 0, 4095).astype(np.uint16)
            pil_img = Image.fromarray(arr, mode='I;16')
            pil_img.save(file_path)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"图像已保存至: {file_path}\n")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SNRApp(root)
    root.mainloop()