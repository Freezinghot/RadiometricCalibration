import os
import re
import time
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
#  全局常量
# ============================================================
SATURATION_VALUE = 4095        # 12bit 过曝阈值
DEFAULT_MEAN_MIN = 800.0       # 均值下限
DEFAULT_MEAN_MAX = 3500.0      # 均值上限
KEY_FEATURES = ('sensor', 'gain', 'tdi', 'channel')
OPT_FEATURES = ('brightness',)
FEATURE_ORDER = ('sensor', 'channel', 'gain', 'tdi', 'brightness')


# ============================================================
#  图像 / 系数 读写辅助
# ============================================================
def imread(path):
    try:
        import tifffile
        img = tifffile.imread(path)
    except ImportError:
        img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def imwrite(path, data):
    try:
        import tifffile
        tifffile.imwrite(path, data)
    except ImportError:
        Image.fromarray(data).save(path)


def load_coefficients(coeff_file):
    df = pd.read_csv(coeff_file)
    if 'Gain' not in df.columns or 'Offset' not in df.columns:
        raise ValueError("系数文件必须包含 'Gain' 和 'Offset' 列。")
    G = df['Gain'].values.astype(np.float64)
    O = df['Offset'].values.astype(np.float64)
    return G, O


# ============================================================
#  坏像元列解析
# ============================================================
def parse_bad_columns(text, n_cols):
    """
    解析坏列字符串，例如 "0,3,5-10,100-110"，列号 0 基。
    返回: (valid_mask, excluded_indices)
    """
    valid = np.ones(n_cols, dtype=bool)
    if not text or not text.strip():
        return valid, np.array([], dtype=int)

    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            if '-' in part:
                a_str, b_str = part.split('-', 1)
                a = int(a_str.strip())
                b = int(b_str.strip())
                if a > b:
                    a, b = b, a
                a = max(0, a)
                b = min(n_cols - 1, b)
                if a <= b:
                    valid[a:b + 1] = False
            else:
                c = int(part)
                if 0 <= c < n_cols:
                    valid[c] = False
                # 列号 >= n_cols 时静默忽略（保持原行为）
        except ValueError:
            raise ValueError(f"坏列格式无法解析: '{part}'")

    excluded = np.where(~valid)[0]
    return valid, excluded


def format_ranges(indices):
    """0 基索引数组 -> "0,3-5,100"（0 基显示）"""
    if len(indices) == 0:
        return ""
    idx = sorted(int(i) for i in indices)
    ranges = []
    start = prev = idx[0]
    for x in idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = x
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


# ============================================================
#  文件收集
# ============================================================
def collect_images(folder, recursive):
    exts = ('.tif', '.tiff')
    files = []
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and fn.lower().endswith(exts):
                files.append(p)
    return sorted(files)


# ============================================================
#  特征提取（文件名优先，目录补充）
# ============================================================
_RE_SENSOR = re.compile(r'CMOS[_\s]?([A-Za-z])(?=[_\.]|$)')
_RE_GAIN_ASCII = re.compile(r'GAIN[_\s]?(\d+(?:\.\d+)?)', re.IGNORECASE)
_RE_TDI = re.compile(r'TDI[_\s]?(\d+)', re.IGNORECASE)
_RE_BRIGHT_ASCII = re.compile(r'BRIGHTNESS[_\s]?(\d+(?:\.\d+)?)', re.IGNORECASE)
_RE_CHANNEL = re.compile(r'(?:^|_)([A-Za-z]\d+)(?=_|$)')
_RE_GAIN_CN = re.compile(r'增益[_\s]?(\d+(?:\.\d+)?)')
_RE_BRIGHT_CN = re.compile(r'亮度[_\s]?(\d+(?:\.\d+)?)')
_RE_FOLDER_SENSOR_A = re.compile(r'^([A-Za-z])_CMOS$', re.IGNORECASE)
_RE_FOLDER_SENSOR_B = re.compile(r'^CMOS_([A-Za-z])$', re.IGNORECASE)


def extract_features(image_path):
    norm = image_path.replace('\\', '/')
    parts = [p for p in norm.split('/') if p]
    if not parts:
        return {}

    filename = parts[-1]
    stem = os.path.splitext(filename)[0]
    folders = parts[:-1]

    features = {}
    search_str = stem.replace('-', '_')

    m = _RE_SENSOR.search(search_str)
    if m:
        features['sensor'] = f'CMOS{m.group(1).upper()}'

    m = _RE_GAIN_ASCII.search(search_str)
    if m:
        features['gain'] = f'GAIN{m.group(1)}'

    m = _RE_TDI.search(search_str)
    if m:
        features['tdi'] = f'TDI{m.group(1)}'

    m = _RE_BRIGHT_ASCII.search(search_str)
    if m:
        features['brightness'] = f'BRIGHTNESS{m.group(1)}'

    for cm in _RE_CHANNEL.finditer(search_str):
        tok = cm.group(1).upper()
        if tok.startswith(('GAIN', 'TDI', 'CMOS')):
            continue
        if re.fullmatch(r'[A-Z]\d+', tok) and len(tok) <= 4:
            features['channel'] = tok
            break

    for folder in folders:
        if 'sensor' not in features:
            m = _RE_FOLDER_SENSOR_A.fullmatch(folder)
            if m:
                features['sensor'] = f'CMOS{m.group(1).upper()}'
            else:
                m = _RE_FOLDER_SENSOR_B.fullmatch(folder)
                if m:
                    features['sensor'] = f'CMOS{m.group(1).upper()}'

        if 'gain' not in features:
            m = _RE_GAIN_CN.search(folder)
            if m:
                features['gain'] = f'GAIN{m.group(1)}'

        if 'tdi' not in features:
            m = _RE_TDI.search(folder)
            if m:
                features['tdi'] = f'TDI{m.group(1)}'

        if 'brightness' not in features:
            m = _RE_BRIGHT_CN.search(folder)
            if m:
                features['brightness'] = f'BRIGHTNESS{m.group(1)}'

    return features


# ============================================================
#  系数 token 规范化 / 匹配
# ============================================================
def normalize_token(s):
    return re.sub(r'[_\-\s\.]', '', s).upper()


def coeff_candidate_tokens(filename):
    stem = os.path.splitext(filename)[0]
    parts = [p for p in re.split(r'[_\-\s]+', stem) if p]
    norm_parts = [normalize_token(p) for p in parts]

    cands = set(norm_parts)
    n = len(norm_parts)
    for i in range(n - 1):
        cands.add(norm_parts[i] + norm_parts[i + 1])
    for i in range(n - 2):
        cands.add(norm_parts[i] + norm_parts[i + 1] + norm_parts[i + 2])
    return cands


def build_coeff_index(coeff_dir):
    index = []
    if not os.path.isdir(coeff_dir):
        return index
    for dirpath, _, filenames in os.walk(coeff_dir):
        for fn in filenames:
            if fn.lower().endswith('.csv'):
                full = os.path.join(dirpath, fn)
                index.append((full, coeff_candidate_tokens(fn)))
    return index


def find_coeff(image_path, coeff_index):
    """
    返回 (coeff_path_or_None, features_dict, display_tokens_list)
    严格模式：图像中提取到的每一个关键特征(sensor/gain/tdi/channel)
              都必须在系数文件名中找到，否则视为未匹配。
    """
    features = extract_features(image_path)
    display_tokens = [features[k] for k in FEATURE_ORDER if k in features]

    if not features or not coeff_index:
        return None, features, display_tokens

    key_norms = [normalize_token(features[k]) for k in KEY_FEATURES if k in features]
    opt_norms = [normalize_token(features[k]) for k in OPT_FEATURES if k in features]

    if not key_norms and not opt_norms:
        return None, features, display_tokens

    n_key = len(key_norms)

    best = None
    for path, cand_set in coeff_index:
        km = sum(1 for t in key_norms if t in cand_set)

        # ★ 关键改动：所有关键特征必须全部命中
        if km < n_key:
            continue

        om = sum(1 for t in opt_norms if t in cand_set)
        score = km * 10 + om

        if best is None or score > best[0] or \
           (score == best[0] and len(path) < len(best[1])):
            best = (score, path, km, om)

    if best is None:
        return None, features, display_tokens

    _, best_path, km, om = best
    return best_path, features, display_tokens


# ============================================================
#  GUI 主程序
# ============================================================
class BatchCorrectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("辐射校正批量验证工具")
        self.root.geometry("1220x920")

        self.img_dir = tk.StringVar()
        self.coeff_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.streak_thr = tk.StringVar(value="1.0")
        self.cv_thr = tk.StringVar(value="0.01")
        self.bad_cols = tk.StringVar(value="2134,1329,1127,354,0-50,2100-2142")
        self.mean_min = tk.StringVar(value=str(DEFAULT_MEAN_MIN))
        self.mean_max = tk.StringVar(value=str(DEFAULT_MEAN_MAX))
        self.recursive = tk.BooleanVar(value=True)
        self.save_corr = tk.BooleanVar(value=False)
        self.check_sat = tk.BooleanVar(value=True)
        self.check_mean = tk.BooleanVar(value=True)

        self.stop_flag = threading.Event()
        self.msg_queue = queue.Queue()
        self.error_list = []
        self.all_results = []
        self.running = False
        self._err_dirty = False

        self._build_ui()
        self._poll_queue()

    # --------------------------------------------------------
    def _build_ui(self):
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        # ---------- 输入 / 输出 ----------
        f_in = ttk.LabelFrame(self.root, text="输入 / 输出", padding=8)
        f_in.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(f_in, text="待校正图像文件夹:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Entry(f_in, textvariable=self.img_dir).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(f_in, text="浏览...", command=self.browse_img_dir).grid(row=0, column=2)

        ttk.Label(f_in, text="系数文件文件夹:").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Entry(f_in, textvariable=self.coeff_dir).grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Button(f_in, text="浏览...", command=self.browse_coeff_dir).grid(row=1, column=2)
        ttk.Label(f_in, text="（留空 = 与图像文件夹相同）",
                  foreground="gray").grid(row=1, column=3, sticky=tk.W, padx=5)

        ttk.Label(f_in, text="输出文件夹(可选):").grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Entry(f_in, textvariable=self.out_dir).grid(row=2, column=1, sticky=tk.EW, padx=5)
        ttk.Button(f_in, text="浏览...", command=self.browse_out_dir).grid(row=2, column=2)
        ttk.Label(f_in, text="（勾选保存时若为空则默认为 图像目录/corrected）",
                  foreground="gray").grid(row=2, column=3, sticky=tk.W, padx=5)

        f_opt = ttk.Frame(f_in)
        f_opt.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        ttk.Checkbutton(f_opt, text="递归子文件夹", variable=self.recursive).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(f_opt, text="保存校正后图像", variable=self.save_corr).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(f_opt,
                        text=f"有效列过曝检查 (=={SATURATION_VALUE} 则跳过)",
                        variable=self.check_sat).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(f_opt, text="均值范围检查", variable=self.check_mean).pack(side=tk.LEFT, padx=5)

        f_in.columnconfigure(1, weight=1)

        # ---------- 阈值 & 坏列 & 均值范围 ----------
        f_thr = ttk.LabelFrame(self.root, text="判定阈值 & 坏像元列 & 均值范围", padding=8)
        f_thr.pack(fill=tk.X, padx=10, pady=5)

        row0 = ttk.Frame(f_thr)
        row0.pack(fill=tk.X)
        ttk.Label(row0, text="校正图像列均值标准差(条带强度) <").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Entry(row0, textvariable=self.streak_thr, width=10).pack(side=tk.LEFT)
        ttk.Label(row0, text="      校正图像全局变异系数(CV) <").pack(side=tk.LEFT, padx=(20, 2))
        ttk.Entry(row0, textvariable=self.cv_thr, width=10).pack(side=tk.LEFT)

        row1 = ttk.Frame(f_thr)
        row1.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row1, text="原始图像有效列均值范围:").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Entry(row1, textvariable=self.mean_min, width=10).pack(side=tk.LEFT)
        ttk.Label(row1, text="~").pack(side=tk.LEFT, padx=2)
        ttk.Entry(row1, textvariable=self.mean_max, width=10).pack(side=tk.LEFT)
        ttk.Label(row1, text="    (不在此区间的图像将跳过)",
                  foreground="gray").pack(side=tk.LEFT, padx=(5, 20))

        row2 = ttk.Frame(f_thr)
        row2.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row2, text="坏像元列 (0基，逗号分隔或范围，如 0,3,5-10,100-110):").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Entry(row2, textvariable=self.bad_cols, width=40).pack(side=tk.LEFT, fill=tk.X,
                                                                expand=True, padx=5)
        ttk.Label(row2, text="留空表示不排除", foreground="gray").pack(side=tk.LEFT, padx=(5, 5))

        # ---------- 按钮 ----------
        f_btn = ttk.Frame(self.root)
        f_btn.pack(fill=tk.X, padx=10, pady=5)

        self.btn_start = ttk.Button(f_btn, text="开始批处理", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(f_btn, text="停止", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Button(f_btn, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(f_btn, text="导出 error_list", command=self.export_errors).pack(side=tk.LEFT, padx=5)
        ttk.Button(f_btn, text="退出", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

        # ---------- 当前文件 & 进度条 ----------
        self.current_var = tk.StringVar(value="等待开始...")
        ttk.Label(self.root, textvariable=self.current_var,
                  anchor=tk.W, foreground="#004080").pack(fill=tk.X, padx=12, pady=(2, 0))
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 5))

        # ---------- 主面板 ----------
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        f_log = ttk.LabelFrame(pane, text="运行日志", padding=5)
        pane.add(f_log, weight=3)
        self.text_log = tk.Text(f_log, wrap=tk.WORD, state=tk.DISABLED,
                                font=("Consolas", 9))
        sb1 = ttk.Scrollbar(f_log, command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=sb1.set)
        sb1.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        f_err = ttk.LabelFrame(pane, text="error_list（未匹配 / 不通过 / 跳过 / 异常）", padding=5)
        pane.add(f_err, weight=2)
        self.text_err = tk.Text(f_err, wrap=tk.WORD, state=tk.DISABLED,
                                font=("Consolas", 9))
        sb2 = ttk.Scrollbar(f_err, command=self.text_err.yview)
        self.text_err.configure(yscrollcommand=sb2.set)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_err.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._refresh_error_list()

    # --------------------------------------------------------
    def browse_img_dir(self):
        d = filedialog.askdirectory(title="选择待校正图像文件夹")
        if d:
            self.img_dir.set(d)

    def browse_coeff_dir(self):
        d = filedialog.askdirectory(title="选择系数文件文件夹")
        if d:
            self.coeff_dir.set(d)

    def browse_out_dir(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.out_dir.set(d)

    # --------------------------------------------------------
    def _log(self, text):
        self.msg_queue.put(('log', text))

    def _append_log(self, text):
        self.text_log.config(state=tk.NORMAL)
        self.text_log.insert(tk.END, text + "\n")
        self.text_log.see(tk.END)
        self.text_log.config(state=tk.DISABLED)

    def clear_log(self):
        self.text_log.config(state=tk.NORMAL)
        self.text_log.delete(1.0, tk.END)
        self.text_log.config(state=tk.DISABLED)

    def _refresh_error_list(self):
        self.text_err.config(state=tk.NORMAL)
        self.text_err.delete(1.0, tk.END)
        if not self.error_list:
            self.text_err.insert(tk.END, "（暂无）")
        else:
            for i, r in enumerate(self.error_list, 1):
                self.text_err.insert(tk.END, f"{i}. {r['image_name']}\n")
                self.text_err.insert(tk.END, f"   系数: {r['coeff_name'] or '(未匹配)'}\n")
                s, c = r['streak'], r['cv']
                s_str = f"{s:.6f}" if s == s else "NaN"
                c_str = f"{c:.6f}" if c == c else "NaN"
                self.text_err.insert(tk.END, f"   条带强度: {s_str}   CV: {c_str}\n")
                self.text_err.insert(tk.END, f"   原因: {r['reason']}\n\n")
        self.text_err.config(state=tk.DISABLED)

    # --------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == 'log':
                    self._append_log(payload)
                elif kind == 'progress':
                    self.progress['value'] = payload
                elif kind == 'status':
                    self.status_var.set(payload)
                elif kind == 'current':
                    self.current_var.set(payload)
                elif kind == 'errorlist':
                    self._err_dirty = True
                elif kind == 'done':
                    self._on_done()
        except queue.Empty:
            pass

        if self._err_dirty:
            self._refresh_error_list()
            self._err_dirty = False

        self.root.after(100, self._poll_queue)

    # --------------------------------------------------------
    def start(self):
        if self.running:
            return

        img_dir = self.img_dir.get().strip()
        if not img_dir or not os.path.isdir(img_dir):
            messagebox.showerror("错误", "请选择有效的待校正图像文件夹。")
            return

        coeff_dir = self.coeff_dir.get().strip() or img_dir
        if not os.path.isdir(coeff_dir):
            messagebox.showerror("错误", "系数文件文件夹无效。")
            return

        try:
            streak_thr = float(self.streak_thr.get())
            cv_thr = float(self.cv_thr.get())
        except ValueError:
            messagebox.showerror("错误", "阈值必须是数字。")
            return

        check_mean = self.check_mean.get()
        mean_min = mean_max = None
        if check_mean:
            try:
                mean_min = float(self.mean_min.get())
                mean_max = float(self.mean_max.get())
                if mean_min >= mean_max:
                    raise ValueError
            except ValueError:
                messagebox.showerror("错误", "均值范围无效（需满足 最小值 < 最大值）。")
                return

        bad_cols_text = self.bad_cols.get().strip()

        out_dir = self.out_dir.get().strip()
        save_corr = self.save_corr.get()
        if save_corr and not out_dir:
            out_dir = os.path.join(img_dir, "corrected")

        images = collect_images(img_dir, self.recursive.get())
        if not images:
            messagebox.showwarning("提示", "该文件夹下未找到 .tif / .tiff 图像。")
            return

        self.stop_flag.clear()
        self.error_list = []
        self.all_results = []
        self._refresh_error_list()
        self.clear_log()
        self.progress['maximum'] = len(images)
        self.progress['value'] = 0
        self.current_var.set("准备中...")
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

        t = threading.Thread(
            target=self._worker,
            args=(images, img_dir, coeff_dir, streak_thr, cv_thr,
                  bad_cols_text, save_corr, out_dir, self.check_sat.get(),
                  check_mean, mean_min, mean_max),
            daemon=True
        )
        t.start()

    def stop(self):
        if self.running:
            self.stop_flag.set()
            self.status_var.set("正在停止...")

    def _on_done(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set(f"完成 —— error_list 共 {len(self.error_list)} 条")
        self._refresh_error_list()

    # --------------------------------------------------------
    def _worker(self, images, img_root, coeff_dir, streak_thr, cv_thr,
                bad_cols_text, save_corr, out_dir, check_sat,
                check_mean, mean_min, mean_max):

        total = len(images)
        ok_cnt = fail_cnt = err_cnt = no_coeff_cnt = skip_cnt = 0
        skip_mean_cnt = 0
        t0 = time.time()

        self._log("=" * 80)
        self._log(f"开始批处理：共 {total} 个 TIF 图像")
        self._log(f"图像目录 : {img_root}")
        self._log(f"系数目录 : {coeff_dir}")
        self._log(f"判定阈值 : 条带强度 < {streak_thr}   CV < {cv_thr}")
        self._log(f"坏像元列 : {bad_cols_text if bad_cols_text else '(无)'}  (0基，不计入统计)")
        self._log(f"过曝检查 : {'开启' if check_sat else '关闭'}"
                  f"  (有效列中出现 =={SATURATION_VALUE} 即跳过)")
        if check_mean:
            self._log(f"均值检查 : 开启  (有效列均值须在 [{mean_min}, {mean_max}] 之间，"
                      f"否则跳过)")
        else:
            self._log(f"均值检查 : 关闭")
        self._log("-" * 80)
        self._log("正在构建系数文件索引...")
        coeff_index = build_coeff_index(coeff_dir)
        self._log(f"共找到 {len(coeff_index)} 个系数 CSV 文件")
        self._log("=" * 80)

        for idx, img_path in enumerate(images, 1):
            if self.stop_flag.is_set():
                self._log(f"\n*** 用户中止，已处理 {idx - 1}/{total} ***")
                break

            name = os.path.basename(img_path)
            self._log(f"[{idx}/{total}] {name}")
            self.msg_queue.put(('status', f"处理中 {idx}/{total}: {name}"))
            self.msg_queue.put(('current', f"[{idx}/{total}] {name}  →  匹配系数中..."))

            coeff_file = None
            coeff_found = False
            img_raw = None
            img_corr = None
            img_valid = None
            raw_dtype = None

            try:
                # ---------- 1. 匹配系数 ----------
                coeff_file, features, tokens = find_coeff(img_path, coeff_index)
                if features:
                    feat_str = " | ".join(f"{k}={v}" for k, v in features.items())
                else:
                    feat_str = "(无有效特征)"
                self._log(f"   特征 : {feat_str}")

                if coeff_file is None:
                    no_coeff_cnt += 1
                    tried = ", ".join(tokens) if tokens else "(无)"
                    self._log(f"   系数 : ✗ 未找到匹配文件  (特征: {tried})")
                    self._log(f"   结果 : ✗ 异常（缺少系数文件）")
                    rec = {
                        'image': img_path,
                        'image_name': name,
                        'coeff': '',
                        'coeff_name': '',
                        'streak': float('nan'),
                        'cv': float('nan'),
                        'reason': f'未找到匹配的系数文件 (特征: {tried})',
                    }
                    self.error_list.append(rec)
                    self.msg_queue.put(('errorlist', None))
                    self.msg_queue.put(('current',
                        f"[{idx}/{total}] {name}  →  系数:✗  验证:✗ (无系数)"))
                    self.msg_queue.put(('progress', idx))
                    self._log("")
                    continue

                coeff_found = True
                coeff_name = os.path.basename(coeff_file)
                self._log(f"   系数 : ✓ {coeff_name}")
                self.msg_queue.put(('current',
                    f"[{idx}/{total}] {name}  →  系数:✓  读取图像中..."))

                # ---------- 2. 读取图像 ----------
                arr = imread(img_path)
                raw_dtype = arr.dtype
                if arr.ndim != 2:
                    raise ValueError(f"仅支持二维单波段图像，当前维度 {arr.ndim}")
                img_raw = arr.astype(np.float64)
                del arr
                K, N = img_raw.shape

                # ---------- 3. 解析坏列 ----------
                valid_mask, excluded_idx = parse_bad_columns(bad_cols_text, N)
                n_valid = int(valid_mask.sum())
                if n_valid < 2:
                    raise ValueError(f"有效列数不足（{n_valid}），请检查坏像元列设置")

                if len(excluded_idx) > 0:
                    self._log(f"   坏列 : 排除 {len(excluded_idx)} 列 -> "
                              f"{format_ranges(excluded_idx)}  (共 {N} 列，有效 {n_valid} 列)")

                # ---------- 4. 过曝检查（只看有效列） ----------
                if check_sat:
                    raw_valid = img_raw[:, valid_mask]
                    n_over = int(np.count_nonzero(raw_valid == SATURATION_VALUE))
                    del raw_valid
                    if n_over > 0:
                        skip_cnt += 1
                        self._log(f"   过曝 : ✗ 有效列中发现 {n_over} 个过曝像素 "
                                  f"(=={SATURATION_VALUE})")
                        self._log(f"   结果 : ⊘ 跳过（不参与校正验证）")
                        rec = {
                            'image': img_path,
                            'image_name': name,
                            'coeff': coeff_file,
                            'coeff_name': coeff_name,
                            'streak': float('nan'),
                            'cv': float('nan'),
                            'reason': f'跳过：除指定列外存在过曝值({SATURATION_VALUE})，'
                                      f'共 {n_over} 个像素',
                        }
                        self.error_list.append(rec)
                        self.msg_queue.put(('errorlist', None))
                        self.msg_queue.put(('current',
                            f"[{idx}/{total}] {name}  →  系数:✓  验证:⊘ 跳过(过曝)"))
                        self.msg_queue.put(('progress', idx))
                        self._log("")
                        del img_raw
                        img_raw = None
                        continue

                # ---------- 5. 均值范围检查（只看有效列，原始 DN） ----------
                if check_mean:
                    raw_valid = img_raw[:, valid_mask]
                    raw_mean = float(np.mean(raw_valid))
                    del raw_valid

                    in_range = (mean_min <= raw_mean <= mean_max)
                    mark = "✓" if in_range else "✗"
                    self._log(f"   均值 : 原始有效列均值={raw_mean:.2f}  "
                              f"([{mean_min}, {mean_max}] {mark})")

                    if not in_range:
                        skip_mean_cnt += 1
                        self._log(f"   结果 : ⊘ 跳过（均值不在指定区间）")
                        rec = {
                            'image': img_path,
                            'image_name': name,
                            'coeff': coeff_file,
                            'coeff_name': coeff_name,
                            'streak': float('nan'),
                            'cv': float('nan'),
                            'reason': f'跳过：原始图像有效列均值 {raw_mean:.2f} 不在 '
                                      f'[{mean_min}, {mean_max}] 区间内',
                        }
                        self.error_list.append(rec)
                        self.msg_queue.put(('errorlist', None))
                        self.msg_queue.put(('current',
                            f"[{idx}/{total}] {name}  →  系数:✓  验证:⊘ 跳过(均值)"))
                        self.msg_queue.put(('progress', idx))
                        self._log("")
                        del img_raw
                        img_raw = None
                        continue

                # ---------- 6. 校正 ----------
                self.msg_queue.put(('current',
                    f"[{idx}/{total}] {name}  →  系数:✓  校正验证中..."))
                G, O = load_coefficients(coeff_file)
                if len(G) != N:
                    raise ValueError(f"系数长度({len(G)}) 与图像列数({N}) 不匹配")

                img_corr = img_raw * G + O

                # ---------- 7. 指标（只取有效列） ----------
                img_valid = img_corr[:, valid_mask]
                col_mean = np.mean(img_valid, axis=0)
                streak = float(np.std(col_mean, ddof=1)) if col_mean.size > 1 else 0.0
                gmean = float(np.mean(img_valid))
                gstd = float(np.std(img_valid, ddof=1))
                cv = gstd / gmean if gmean != 0 else float('nan')

                # ---------- 8. 判定 ----------
                streak_ok = (streak < streak_thr)
                cv_ok = (cv < cv_thr)
                passed = streak_ok and cv_ok

                reasons = []
                if not streak_ok:
                    reasons.append(f"条带强度 {streak:.6f} >= {streak_thr}")
                if not cv_ok:
                    reasons.append(f"CV {cv:.6f} >= {cv_thr}")

                s_mark = "✓" if streak_ok else "✗"
                c_mark = "✓" if cv_ok else "✗"
                self._log(f"   验证 : 条带强度={streak:.6f} (<{streak_thr} {s_mark})  |  "
                          f"CV={cv:.6f} (<{cv_thr} {c_mark})")

                rec = {
                    'image': img_path,
                    'image_name': name,
                    'coeff': coeff_file,
                    'coeff_name': coeff_name,
                    'streak': streak,
                    'cv': cv,
                    'reason': ' ; '.join(reasons) if reasons else '',
                }
                self.all_results.append(rec)

                if passed:
                    ok_cnt += 1
                    self._log(f"   结果 : ✓ 通过")
                else:
                    fail_cnt += 1
                    self.error_list.append(rec)
                    self._log(f"   结果 : ✗ 不通过  ({rec['reason']})")
                    self.msg_queue.put(('errorlist', None))

                # ---------- 9. 可选保存 ----------
                if save_corr and out_dir:
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(
                            out_dir, os.path.splitext(name)[0] + '_corr.tif')
                        if np.issubdtype(raw_dtype, np.integer):
                            info = np.iinfo(raw_dtype)
                            data = np.clip(np.round(img_corr),
                                           info.min, info.max).astype(raw_dtype)
                        else:
                            data = img_corr.astype(np.float32)
                        imwrite(out_path, data)
                        self._log(f"   保存 : {out_path}")
                    except Exception as e:
                        self._log(f"   保存失败: {e}")

                self.msg_queue.put(('current',
                    f"[{idx}/{total}] {name}  →  系数:✓  验证:{'✓ 通过' if passed else '✗ 不通过'}"))

            except Exception as e:
                err_cnt += 1
                self._log(f"   结果 : ✗ 异常 - {e}")
                rec = {
                    'image': img_path,
                    'image_name': name,
                    'coeff': coeff_file or '',
                    'coeff_name': os.path.basename(coeff_file) if coeff_file else '',
                    'streak': float('nan'),
                    'cv': float('nan'),
                    'reason': f'处理异常: {e}',
                }
                self.error_list.append(rec)
                self.msg_queue.put(('errorlist', None))
                self.msg_queue.put(('current',
                    f"[{idx}/{total}] {name}  →  系数:{'✓' if coeff_found else '✗'}  验证:✗ 异常"))

            finally:
                try:
                    if img_raw is not None:
                        del img_raw
                    if img_corr is not None:
                        del img_corr
                    if img_valid is not None:
                        del img_valid
                except Exception:
                    pass

            self.msg_queue.put(('progress', idx))
            self._log("")

        # ---------------- 汇总 ----------------
        elapsed = time.time() - t0
        self._log("=" * 80)
        self._log("批处理结束")
        self._log(f"  总文件数         : {total}")
        self._log(f"  通过             : {ok_cnt}")
        self._log(f"  指标不通过       : {fail_cnt}")
        self._log(f"  过曝跳过         : {skip_cnt}")
        self._log(f"  均值跳过         : {skip_mean_cnt}")
        self._log(f"  未找到系数文件   : {no_coeff_cnt}")
        self._log(f"  处理异常         : {err_cnt}")
        self._log(f"  耗时             : {elapsed:.1f} 秒")
        self._log(f"  error_list 条目数: {len(self.error_list)}")
        self._log("-" * 80)

        if self.error_list:
            self._log("error_list（图像 -> 匹配系数 -> 原因）:")
            for i, r in enumerate(self.error_list, 1):
                self._log(f"  {i:>3}. {r['image_name']}")
                self._log(f"       -> 系数: {r['coeff_name'] or '(未匹配)'}")
                self._log(f"       -> 原因: {r['reason']}")
        else:
            self._log("error_list 为空，全部文件均通过验证。")
        self._log("=" * 80)

        self.msg_queue.put(('done', None))

    # --------------------------------------------------------
    def export_errors(self):
        if not self.error_list:
            messagebox.showinfo("提示", "error_list 为空，无需导出。")
            return

        path = filedialog.asksaveasfilename(
            title="导出 error_list",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if not path:
            return

        try:
            if path.lower().endswith('.txt'):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("图像文件\t匹配系数文件\t条带强度\t全局CV\t原因\n")
                    for r in self.error_list:
                        f.write(f"{r['image_name']}\t{r['coeff_name']}\t"
                                f"{r['streak']:.6f}\t{r['cv']:.6f}\t{r['reason']}\n")
            else:
                df = pd.DataFrame(self.error_list)
                df.to_csv(path, index=False, encoding='utf-8-sig')

            messagebox.showinfo("导出成功", f"error_list 已保存至:\n{path}")
            self.status_var.set(f"error_list 已导出: {path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = BatchCorrectionApp(root)
    root.mainloop()