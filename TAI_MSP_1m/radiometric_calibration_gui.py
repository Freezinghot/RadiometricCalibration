# --*-- conding:utf-8 --*--
# @File  : radiometric_calibration_gui.py
# @Author: FH
# @Date  : 2026/8/18
# @Desc  :
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ======================== 核心计算函数 ========================
def read_means_from_csv(file_path):
    df = pd.read_csv(file_path)
    if 'Mean_Value' not in df.columns:
        raise ValueError(f"File {file_path} does not contain column 'Mean_Value'")
    return df['Mean_Value'].astype(float).tolist()


def calculate_multi_point_coeffs_from_means(dark_mean, light_means_list, L_values):
    """
    计算多点相对辐射定标的增益和偏置系数，同时返回每个像元的拟合参数。

    参数：
        dark_mean: np.array, 暗场（L=0）下的像元均值，形状 (N,)，N为像元数。
        light_means_list: list of np.array，每个元素是某个非零辐射亮度下的像元均值，形状均为 (N,)。
        L_values: list of float，包含暗场对应的0以及所有亮场对应的辐射亮度值，
                  长度应为 1 + len(light_means_list)。

    返回：
        G: np.array, 增益系数 (N,)
        O: np.array, 偏置系数 (N,)
        a: np.array, 每个像元线性拟合的斜率 (N,)
        b: np.array, 每个像元线性拟合的截距 (N,)
    """
    all_means = np.vstack([dark_mean] + light_means_list)

    L = np.asarray(L_values, dtype=np.float64)
    L_mean = np.mean(L)
    L_centered = L - L_mean

    mean_centered = all_means - np.mean(all_means, axis=0)

    numerator = np.sum(L_centered[:, np.newaxis] * mean_centered, axis=0)
    denominator = np.sum(L_centered ** 2)
    a = numerator / denominator

    mean_y = np.mean(all_means, axis=0)
    b = mean_y - a * L_mean

    valid = a > 1e-9
    if not np.any(valid):
        raise ValueError("All pixel slopes are near zero, cannot compute reference line.")

    A_ref = np.mean(a[valid])
    B_ref = np.mean(b[valid])

    G = np.ones_like(a)
    O = np.zeros_like(b)
    G[valid] = A_ref / a[valid]
    O[valid] = B_ref - G[valid] * b[valid]
    O[~valid] = B_ref

    return G, O, a, b


def calculate_coeffs_with_dark_subtraction(dark_mean, light_means_list, L_values, saturation_threshold=4000, max_resid_std=3.0):
    """
    基于“先扣暗场再线性拟合”的定标系数计算。
    """
    dark_mean = np.asarray(dark_mean, dtype=np.float64)
    light_means_list = [np.asarray(m, dtype=np.float64) for m in light_means_list]
    L_values = np.asarray(L_values, dtype=np.float64)
    light_arrs = [np.asarray(m) for m in light_means_list]
    N = len(dark_mean)

    all_raw = np.asarray(light_arrs)
    all_sub = all_raw - dark_mean[np.newaxis, :]

    N = dark_mean.shape[0]
    k = np.zeros(N)
    c = np.zeros(N)
    valid = np.ones(N, dtype=bool)

    for i in range(N):
        y = all_sub[:, i]
        L = L_values[1:]
        raw = all_raw[:, i]
        mask = raw < saturation_threshold
        L_clean = L[mask]
        y_clean = y[mask]
        if len(L_clean) < 2:
            valid[i] = False
            continue
        p = np.polyfit(L_clean, y_clean, 1)
        # 残差二次剔除，在点数少时会删光所有点，亮度采样点少时删除此环节
        # resid = y_clean - np.polyval(p, L_clean)
        # std = np.std(resid)
        # mask2 = np.abs(resid) < max_resid_std * std
        # L_final = L_clean[mask2]
        # y_final = y_clean[mask2]
        # if len(L_final) < 2:
        #     valid[i] = False
        #     continue
        # p = np.polyfit(L_final, y_final, 1)
        k[i] = p[0]
        c[i] = p[1]

    valid = k > 1e-9
    if not np.any(valid):
        raise ValueError("All pixel slopes after dark subtraction are near zero.")

    A_ref = np.mean(k[valid])
    B_ref = np.mean(c[valid])

    G = np.ones_like(k)
    O = np.zeros_like(c)

    G[valid] = A_ref / k[valid]
    O[valid] = B_ref - G[valid] * c[valid]

    O[~valid] = B_ref

    O_new = O - G * dark_mean

    return G, O_new, k, c


def calc_fit_r2(all_means, L_vals, a, b):
    N = all_means.shape[1]
    r2 = np.zeros(N)
    for i in range(N):
        y = all_means[:, i]
        y_pred = a[i] * L_vals + b[i]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-12:
            r2[i] = 1.0
        else:
            r2[i] = 1 - ss_res / ss_tot
    return r2


def calculate_coeffs_with_dark_subtraction_v2(dark_mean, light_means_list, L_values,
                                              saturation_threshold=4000):
    light_arrs = [np.asarray(m) for m in light_means_list]
    N = len(dark_mean)
    all_raw = np.vstack([dark_mean] + light_arrs)
    all_sub = all_raw - dark_mean[np.newaxis, :]
    L = np.asarray(L_values, dtype=np.float64)

    k = np.zeros(N)
    c = np.zeros(N)
    valid = np.ones(N, dtype=bool)

    for i in range(N):
        y = all_sub[:, i]
        raw = all_raw[:, i]
        mask = raw < saturation_threshold
        if np.sum(mask) < 2:
            valid[i] = False
            continue
        L_clean = L[mask]
        y_clean = y[mask]
        w = 1.0 / np.sqrt(y_clean + 1)
        p = np.polyfit(L_clean, y_clean, 1, w=w)
        k[i] = p[0]
        c[i] = p[1]

    valid = valid & (k > 1e-9)
    if not np.any(valid):
        raise ValueError("No valid pixels found.")

    A_ref = np.median(k[valid])
    B_ref = np.median(c[valid])
    G = np.ones(N); O = np.zeros(N)
    G[valid] = A_ref / k[valid]
    O[valid] = B_ref - G[valid] * c[valid]
    O[~valid] = B_ref
    return G, O, k, c


# ======================== 应用程序类 ========================
class RadiometricCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Relative Radiometric Calibration Coefficients Calculator")
        self.root.geometry("1100x650")

        self.file_records = []
        self.current_edit_item = None
        self.current_edit_column = None
        self.edit_entry = None

        # 计算结果存储
        self.G = None
        self.O = None
        self.fit_a = None
        self.fit_b = None
        self.fit_r2 = None
        self.all_means = None
        self.L_arr = None

        # 亮度预设映射 { "亮度1": 108.1, "亮度2": 200.5, ... }
        self.brightness_map = {}

        # matplotlib 画布
        self.canvas = None
        self.ax = None
        self.fig = None

        self.create_widgets()
        self._bind_tree_edit_events()

    # ------------------------------------------------------------------
    # 界面构建（左右分栏）
    # ------------------------------------------------------------------
    def create_widgets(self):
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True)

        # ----- 左侧：文件管理区域 -----
        left_frame = ttk.Frame(main_pw)
        main_pw.add(left_frame, weight=1)

        ttk.Label(left_frame,
                  text="Add calibration data files (CSV with 'Mean_Value' column). Double-click 'Radiance' to edit.",
                  padding=5).pack(anchor=tk.W, padx=10, pady=5)

        # 表格
        frame_table = ttk.Frame(left_frame)
        frame_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("File Path", "Radiance")
        self.tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=10)
        self.tree.heading("File Path", text="File Path")
        self.tree.heading("Radiance", text="Radiance")
        self.tree.column("File Path", width=400)
        self.tree.column("Radiance", width=100, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # 按钮区
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Brightness Map", command=self.edit_brightness_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Edit Radiance", command=self.edit_selected_radiance).pack(side=tk.LEFT, padx=5)

        calc_frame = ttk.Frame(left_frame)
        calc_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_calc = ttk.Button(calc_frame, text="Calculate Coefficients", command=self.calculate)
        self.btn_calc.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(calc_frame, text="Save Coefficients...", state=tk.DISABLED, command=self.save_coeffs)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        ttk.Button(calc_frame, text="Exit", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

        # 状态栏（左下方）
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(left_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=2)

        # ----- 右侧：拟合结果显示 -----
        right_frame = ttk.Frame(main_pw)
        main_pw.add(right_frame, weight=1)

        self.stats_text = tk.StringVar(value="Fit accuracy will be shown here after calculation.")
        ttk.Label(right_frame, textvariable=self.stats_text, font=("TkDefaultFont", 10, "bold"),
                  padding=10).pack(anchor=tk.W)

        pixel_frame = ttk.Frame(right_frame)
        pixel_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(pixel_frame, text="Pixel index for fit plot:").pack(side=tk.LEFT)
        self.pixel_entry = ttk.Entry(pixel_frame, width=8)
        self.pixel_entry.pack(side=tk.LEFT, padx=5)
        self.pixel_entry.insert(0, "0")
        self.pixel_entry.bind("<Return>", lambda e: self.update_plot())
        ttk.Button(pixel_frame, text="Plot", command=self.update_plot).pack(side=tk.LEFT, padx=5)

        self.plot_frame = ttk.Frame(right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        if MATPLOTLIB_AVAILABLE:
            self.init_plot_canvas()
        else:
            ttk.Label(self.plot_frame, text="matplotlib not installed. Plot unavailable.",
                      foreground="red").pack(anchor=tk.CENTER, expand=True)

    def init_plot_canvas(self):
        """初始化 matplotlib 画布（空白）"""
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No data yet")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # 表格编辑功能（保持不变）
    # ------------------------------------------------------------------
    def _bind_tree_edit_events(self):
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.tree.bind("<Up>", self.on_arrow_key)
        self.tree.bind("<Down>", self.on_arrow_key)

    def on_tree_double_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.tree.identify_column(event.x)
        item = self.tree.identify_row(event.y)
        if column == "#2" and item:
            self._start_edit(item, column)

    def edit_selected_radiance(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No selection", "Please select a file to edit.")
            return
        if len(selected) > 1:
            messagebox.showwarning("Multiple selection", "Only one file can be edited at a time.")
            return
        self._start_edit(selected[0], "#2")

    def _start_edit(self, item, column):
        if self.edit_entry is not None:
            self._accept_edit()

        bbox = self.tree.bbox(item, column)
        if not bbox:
            return
        x, y, width, height = bbox

        rec = next((r for r in self.file_records if r['item_id'] == item), None)
        if not rec:
            return
        current_val = f"{rec['L']:.4f}"

        self.edit_entry = tk.Entry(self.tree, width=12)
        self.edit_entry.place(x=x, y=y, width=width, height=height)
        self.edit_entry.insert(0, current_val)
        self.edit_entry.select_range(0, tk.END)
        self.edit_entry.focus_set()

        self.edit_entry.bind("<Return>", lambda e: self._accept_edit())
        self.edit_entry.bind("<Escape>", lambda e: self._cancel_edit())
        self.edit_entry.bind("<Up>", self._on_edit_up)
        self.edit_entry.bind("<Down>", self._on_edit_down)
        self.edit_entry.bind("<FocusOut>", self._on_focus_out)

        self.current_edit_item = item
        self.current_edit_column = column

    def _accept_edit(self):
        if self.edit_entry is None:
            return
        new_text = self.edit_entry.get()
        try:
            new_val = float(new_text)
        except ValueError:
            messagebox.showwarning("Invalid input", "Please enter a valid number.")
            self._cancel_edit()
            return

        rec = next((r for r in self.file_records if r['item_id'] == self.current_edit_item), None)
        if rec:
            rec['L'] = new_val
            self.tree.set(self.current_edit_item, "Radiance", f"{new_val:.4f}")

        self._destroy_edit()

    def _cancel_edit(self):
        self._destroy_edit()

    def _destroy_edit(self):
        if self.edit_entry:
            self.edit_entry.destroy()
            self.edit_entry = None
        self.current_edit_item = None
        self.current_edit_column = None

    def _on_focus_out(self, event):
        if self.edit_entry:
            self.root.after(100, self._auto_accept_if_idle)

    def _auto_accept_if_idle(self):
        if self.edit_entry and self.edit_entry.focus_get() != self.edit_entry:
            self._accept_edit()

    def on_arrow_key(self, event):
        if self.edit_entry is None:
            return

    def _on_edit_up(self, event):
        self._accept_edit()
        self._move_edit_row(-1)
        return "break"

    def _on_edit_down(self, event):
        self._accept_edit()
        self._move_edit_row(1)
        return "break"

    def _move_edit_row(self, offset):
        all_items = self.tree.get_children()
        if not all_items:
            return
        prev = self.current_edit_item
        if prev in all_items:
            idx = all_items.index(prev)
            new_idx = (idx + offset) % len(all_items)
        else:
            new_idx = 0
        self._start_edit(all_items[new_idx], "#2")

    # ------------------------------------------------------------------
    # 文件添加/删除
    # ------------------------------------------------------------------
    def add_files(self):
        filenames = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        for f in filenames:
            if any(rec['path'] == f for rec in self.file_records):
                continue

            # 自动匹配亮度值
            key = self._extract_brightness_key(f)
            if key == '暗场':
                L = 0.0  # 暗场固定为0
                auto_note = "暗场→0"
            elif key and key in self.brightness_map:
                L = self.brightness_map[key]
                auto_note = f"{key}→{L}"
            else:
                L = 0.0
                auto_note = "未识别，需手动编辑"

            item_id = self.tree.insert("", tk.END, values=(f, f"{L:.4f}"))
            self.file_records.append({'path': f, 'L': L, 'item_id': item_id})

            if auto_note != "暗场→0" and auto_note.startswith("未识别"):
                self.status_var.set(f"Added {len(filenames)} file(s), some could not be auto-matched. Please edit Radiance manually or set Brightness Map.")
            else:
                self.status_var.set(f"Added {len(filenames)} file(s)")

        if filenames:
            self.status_var.set(f"Added {len(filenames)} file(s)")
        else:
            self.status_var.set("No files selected")

    def remove_files(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No selection", "Please select files to remove.")
            return
        if self.edit_entry and self.current_edit_item in selected:
            self._cancel_edit()
        for item in selected:
            self.tree.delete(item)
            self.file_records = [rec for rec in self.file_records if rec['item_id'] != item]
        self.status_var.set(f"Removed {len(selected)} file(s)")

    # ------------------------------------------------------------------
    # 亮度映射设置
    # ------------------------------------------------------------------
    def edit_brightness_map(self):
        """打开亮度映射编辑对话框"""
        win = tk.Toplevel(self.root)
        win.title("Brightness Map Settings")
        win.geometry("400x500")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="每行输入一个亮度等级，格式：亮度n=数值\n例如：亮度1=108.1\n最多20个，暗场自动识别为0，无需设置。").pack(pady=5)

        text = tk.Text(win, height=20, width=40)
        text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # 预填当前映射
        for key, val in self.brightness_map.items():
            text.insert(tk.END, f"{key}={val}\n")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)

        def save_mapping():
            content = text.get("1.0", tk.END).strip()
            lines = content.splitlines()
            new_map = {}
            error = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if '=' not in line:
                    error = f"无效行: {line}"
                    break
                key, val_str = line.split('=', 1)
                key = key.strip()
                val_str = val_str.strip()
                if not key.startswith('亮度') or not key[2:].isdigit():
                    error = f"键格式错误: {key} (应为 亮度n)"
                    break
                try:
                    val = float(val_str)
                except ValueError:
                    error = f"数值无效: {val_str}"
                    break
                new_map[key] = val

            if error:
                messagebox.showerror("Error", error, parent=win)
                return

            if len(new_map) > 20:
                messagebox.showerror("Error", "最多支持20个亮度等级（不含暗场）", parent=win)
                return

            self.brightness_map = new_map
            self.status_var.set(f"Brightness map updated with {len(new_map)} entries.")
            self._auto_match_all_files()
            win.destroy()

        ttk.Button(btn_frame, text="Save", command=save_mapping).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=5)

    def _extract_brightness_key(self, filepath):
        """从文件名提取亮度标识，返回 '暗场' 或 '亮度n'，未找到返回 None"""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        if '暗场' in basename:
            return '暗场'
        match = re.search(r'亮度(\d+)', basename)
        if match:
            return f'亮度{match.group(1)}'
        return None

    def _auto_match_all_files(self):
        """根据当前映射重新匹配所有已添加文件的 Radiance"""
        for rec in self.file_records:
            key = self._extract_brightness_key(rec['path'])
            if key == '暗场':
                new_L = 0.0
            elif key and key in self.brightness_map:
                new_L = self.brightness_map[key]
            else:
                # 保持原值，若原本是0可能仍需手动编辑，这里不改动
                continue
            rec['L'] = new_L
            self.tree.set(rec['item_id'], "Radiance", f"{new_L:.4f}")

    # ------------------------------------------------------------------
    # 计算定标系数
    # ------------------------------------------------------------------
    def calculate(self):
        if not self.file_records:
            messagebox.showerror("No data", "Please add calibration files first.")
            return

        records = sorted(self.file_records, key=lambda x: x['L'])
        dark_recs = [r for r in records if r['L'] == 0.0]
        if not dark_recs:
            messagebox.showerror("Dark missing", "At least one file with radiance 0 is required as dark frame.")
            return
        if len(dark_recs) > 1:
            messagebox.showwarning("Multiple darks", "Multiple files with radiance 0 detected. Only the first will be used as dark.")
        dark_rec = dark_recs[0]

        light_recs = [r for r in records if r['L'] != 0.0]
        if not light_recs:
            messagebox.showerror("Light missing", "At least one file with non-zero radiance is required.")
            return

        L_set = set(r['L'] for r in records)
        if len(L_set) < 2:
            messagebox.showerror("Insufficient radiance values", "At least two distinct radiance values (including 0) are needed.")
            return

        try:
            dark_mean = np.array(read_means_from_csv(dark_rec['path']))
        except Exception as e:
            messagebox.showerror("Dark read error", f"Cannot read dark file:\n{dark_rec['path']}\nError: {e}")
            return

        light_means = []
        L_vals = [0.0]
        for rec in light_recs:
            try:
                mean_arr = np.array(read_means_from_csv(rec['path']))
                light_means.append(mean_arr)
                L_vals.append(rec['L'])
            except Exception as e:
                messagebox.showerror("Light read error", f"File:\n{rec['path']}\nError: {e}")
                return

        N = len(dark_mean)
        for i, arr in enumerate(light_means):
            if len(arr) != N:
                messagebox.showerror("Size mismatch",
                    f"Dark pixel count ({N}) differs from light file\n{light_recs[i]['path']}\n({len(arr)}).")
                return

        try:
            # G, O, a, b = calculate_multi_point_coeffs_from_means(dark_mean, light_means, L_vals)
            G, O, a, b = calculate_coeffs_with_dark_subtraction(dark_mean, light_means, L_vals)
        except Exception as e:
            messagebox.showerror("Calculation failed", f"Error: {e}")
            return

        all_means = np.vstack([dark_mean] + light_means)
        all_sub = all_means - dark_mean[np.newaxis, :]
        L_arr = np.array(L_vals)
        r2 = calc_fit_r2(all_sub, L_arr, a, b)

        self.G = G
        self.O = O
        self.fit_a = a
        self.fit_b = b
        self.fit_r2 = r2
        self.all_means = all_means
        self.L_arr = L_arr

        self.btn_save.config(state=tk.NORMAL)

        self.all_sub = all_means - dark_mean[np.newaxis, :]

        stats = (
            f"Calibration completed.\n"
            f"Pixels: {N}\n"
            f"Gain range: [{G.min():.4f}, {G.max():.4f}]\n"
            f"Offset range: [{O.min():.4f}, {O.max():.4f}]\n"
            f"R²   min: {r2.min():.6f}   max: {r2.max():.6f}   mean: {r2.mean():.6f}"
        )
        self.stats_text.set(stats)

        self.update_plot(default_idx=0)

    # ------------------------------------------------------------------
    # 绘制指定像元的拟合曲线
    # ------------------------------------------------------------------
    def update_plot(self, default_idx=None):
        if not MATPLOTLIB_AVAILABLE or self.all_means is None:
            if default_idx is not None:
                messagebox.showinfo("No plot", "Calculation data is not available.")
            return

        if default_idx is not None:
            idx_str = str(default_idx)
            self.pixel_entry.delete(0, tk.END)
            self.pixel_entry.insert(0, idx_str)
        else:
            idx_str = self.pixel_entry.get().strip()
        try:
            pix_idx = int(idx_str)
        except ValueError:
            messagebox.showerror("Invalid index", "Please enter a valid integer pixel index.")
            return

        if pix_idx < 0 or pix_idx >= len(self.fit_a):
            messagebox.showerror("Index out of range", f"Pixel index must be between 0 and {len(self.fit_a)-1}.")
            return

        self.ax.clear()

        i = pix_idx
        y_real = self.all_sub[:, i]
        x_vals = self.L_arr
        y_fit = self.fit_a[i] * x_vals + self.fit_b[i]

        self.ax.scatter(x_vals, y_real, color='blue', label='Measured mean')
        self.ax.plot(x_vals, y_fit, 'r-', label=f'Fit (R² = {self.fit_r2[i]:.6f})')
        self.ax.set_xlabel("Radiance L")
        self.ax.set_ylabel("Pixel mean")
        self.ax.set_title(f"Pixel #{pix_idx} linear fit")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

    # ------------------------------------------------------------------
    # 保存系数
    # ------------------------------------------------------------------
    def save_coeffs(self):
        if self.G is None or self.O is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save calibration coefficients"
        )
        if not file_path:
            return
        try:
            df = pd.DataFrame({"Gain": self.G, "Offset": self.O})
            df.to_csv(file_path, index=False)
            self.status_var.set(f"Coefficients saved to {file_path}")
            messagebox.showinfo("Save successful", f"Coefficients saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ======================== 主入口 ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = RadiometricCalibrationApp(root)
    root.mainloop()