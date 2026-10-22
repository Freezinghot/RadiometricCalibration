# --*-- conding:utf-8 --*--
# @File  : csv2npz.py
# @Author: FH
# @Date  : 2026/10/22
# @Desc  :
# -*- coding: utf-8 -*-
"""
辐射定标数据整合工具（GUI）

功能：
  遍历指定文件夹内的 *CMOS<字母>_<B数字|P>_GAIN<数值>_TDI<数值>.csv 文件，
  按 (CMOS, 波段) 分组整合为 msp_relative_cmos<编号>_<波段>.npz。
  波段支持：
    CMOSA_B1_GAIN2.5_TDI2.csv  -> msp_relative_cmos1_b1.npz   (标签 gain2.5_tdi2)
    CMOSB_P_GAIN2_TDI2.csv     -> msp_relative_cmos2_p.npz    (标签 gain2_tdi2)

NPZ 内部结构：
  每个 npz 中 key = "gain<gain值>_tdi<tdi值>"（例如 gain2.5_tdi2）
  value = numpy 结构化数组，字段为 'Gain' 与 'Offset'
  读取示例：
      d = np.load("msp_relative_cmos1_b1.npz")
      arr = d["gain2.5_tdi2"]
      gains, offsets = arr["Gain"], arr["Offset"]

依赖：numpy（标准库外的唯一依赖）
"""

import os
import re
import csv
import queue
import threading

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


# ------------------------------------------------------------------
# 文件名解析
# ------------------------------------------------------------------
FILENAME_RE = re.compile(
    r'^CMOS([A-Za-z])[_-]?(B\d+|P)[_-]?'
    r'GAIN[_-]?([0-9]+(?:\.[0-9]+)?)[_-]?'
    r'TDI[_-]?([0-9]+(?:\.[0-9]+)?)\.csv$',
    re.IGNORECASE,
)

TEXT_ENCODINGS = ('utf-8-sig', 'gbk', 'utf-8', 'latin-1')


def format_number_str(s: str) -> str:
    """'2.5'->'2.5', '2.0'->'2', '2'->'2'"""
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
        return s if s else '0'
    return s


def parse_filename(fname: str):
    """返回 (cmos编号, 波段小写, 标签) 或 None"""
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    letter, band, gain_s, tdi_s = m.groups()
    cmos = ord(letter.upper()) - ord('A') + 1
    band = band.lower()                       # 'B1'->'b1', 'P'->'p'
    label = f"gain{format_number_str(gain_s)}_tdi{format_number_str(tdi_s)}"
    return cmos, band, label


# ------------------------------------------------------------------
# CSV 读取
# ------------------------------------------------------------------
def read_gain_offset(path: str):
    """返回 (gains, offsets, skipped_rows)"""
    rows = None
    for enc in TEXT_ENCODINGS:
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        raise IOError(f"无法按任何已知编码读取: {path}")

    header_idx = None
    gi = oi = None
    for i, row in enumerate(rows):
        lowered = [c.strip().lower() for c in row]
        if 'gain' in lowered and 'offset' in lowered:
            header_idx = i
            gi = lowered.index('gain')
            oi = lowered.index('offset')
            break
    if header_idx is None:
        raise ValueError("未找到包含 Gain / Offset 的表头行")

    gains, offsets = [], []
    skipped = 0
    for row in rows[header_idx + 1:]:
        if not row or all(c.strip() == '' for c in row):
            continue
        if len(row) <= max(gi, oi):
            skipped += 1
            continue
        try:
            g = float(row[gi].strip())
            o = float(row[oi].strip())
        except ValueError:
            skipped += 1
            continue
        gains.append(g)
        offsets.append(o)

    return (np.asarray(gains, dtype=np.float64),
            np.asarray(offsets, dtype=np.float64),
            skipped)


# ------------------------------------------------------------------
# 核心处理流程
# ------------------------------------------------------------------
def collect_files(folder: str, recursive: bool):
    result = []
    if recursive:
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                if FILENAME_RE.match(f):
                    result.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and FILENAME_RE.match(f):
                result.append(p)
    result.sort()
    return result


def run_process(input_dir, output_dir, recursive, log, progress_cb):
    if not os.path.isdir(input_dir):
        raise ValueError("输入文件夹不存在")

    files = collect_files(input_dir, recursive)
    if not files:
        log("未在输入文件夹中找到匹配 “CMOSx_<Bn|P>_GAIN..._TDI....csv” 的文件。")
        return

    log(f"共找到 {len(files)} 个待处理 CSV 文件。")

    # (cmos, band) -> { label: (gains, offsets) }
    groups = {}
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        parsed = parse_filename(fname)
        if parsed is None:
            log(f"[跳过] 无法解析文件名: {fname}")
            progress_cb((i + 1) / len(files))
            continue
        cmos, band, label = parsed

        try:
            gains, offsets, skipped = read_gain_offset(path)
        except Exception as e:
            log(f"[错误] 读取 {fname} 失败: {e}")
            progress_cb((i + 1) / len(files))
            continue

        if gains.size == 0:
            log(f"[警告] {fname} 中没有有效数据行，已跳过。")
            progress_cb((i + 1) / len(files))
            continue

        if skipped:
            log(f"[警告] {fname} 中有 {skipped} 行无法解析，已忽略。")

        bucket = groups.setdefault((cmos, band), {})
        if label in bucket:
            log(f"[警告] {fname} 的标签 “{label}” 与先前文件重复，将被覆盖。")
        bucket[label] = (gains, offsets)

        log(f"[读取] {fname}  →  cmos{cmos}/{band}/{label},  {gains.size} 行")
        progress_cb((i + 1) / len(files))

    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for (cmos, band), items in sorted(groups.items()):
        out_name = f"msp_relative_cmos{cmos}_{band}.npz"
        out_path = os.path.join(output_dir, out_name)

        payload = {}
        for label, (gains, offsets) in items.items():
            arr = np.vstack([gains, offsets]).astype(np.float64, copy=False)
            payload[label] = arr

        np.savez_compressed(out_path, **payload)
        written += 1
        log(f"[写出] {out_name}  ←  {len(payload)} 组: "
            f"{', '.join(sorted(payload.keys()))}")

    log(f"完成：共写出 {written} 个 npz 文件。")


# ------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------
class CalibApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("辐射定标数据整合工具")
        self.geometry("820x600")
        self.minsize(720, 500)

        self.msg_queue = queue.Queue()
        self.worker = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        path_frame = ttk.LabelFrame(self, text="路径设置")
        path_frame.pack(fill='x', padx=10, pady=(10, 4))

        ttk.Label(path_frame, text="输入文件夹:").grid(
            row=0, column=0, sticky='w', **pad)
        ttk.Entry(path_frame, textvariable=self.input_var).grid(
            row=0, column=1, sticky='ew', **pad)
        ttk.Button(path_frame, text="浏览...",
                   command=self._choose_input).grid(row=0, column=2, **pad)

        ttk.Label(path_frame, text="输出文件夹:").grid(
            row=1, column=0, sticky='w', **pad)
        ttk.Entry(path_frame, textvariable=self.output_var).grid(
            row=1, column=1, sticky='ew', **pad)
        ttk.Button(path_frame, text="浏览...",
                   command=self._choose_output).grid(row=1, column=2, **pad)

        ttk.Checkbutton(path_frame, text="包含子文件夹（递归遍历）",
                        variable=self.recursive_var).grid(
            row=2, column=1, sticky='w', **pad)

        path_frame.columnconfigure(1, weight=1)

        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=10, pady=4)
        self.start_btn = ttk.Button(bar, text="开始处理",
                                    command=self._start)
        self.start_btn.pack(side='left')
        self.progress = ttk.Progressbar(bar, maximum=100, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True, padx=8)

        log_frame = ttk.LabelFrame(self, text="处理日志")
        log_frame.pack(fill='both', expand=True, padx=10, pady=(4, 10))
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=18, state='disabled',
            font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    def _choose_input(self):
        d = filedialog.askdirectory(title="选择输入文件夹")
        if d:
            self.input_var.set(d)
            if not self.output_var.get():
                self.output_var.set(os.path.join(d, "npz_output"))

    def _choose_output(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.output_var.set(d)

    # ------------------------------------------------------------------
    def _append_log(self, text):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', text + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _poll_queue(self):
        while True:
            try:
                kind, payload = self.msg_queue.get_nowait()
            except queue.Empty:
                break

            if kind == 'log':
                self._append_log(payload)
            elif kind == 'progress':
                self.progress['value'] = payload * 100
            elif kind == 'done':
                self.start_btn.configure(state='normal')
                if payload:
                    self.progress['value'] = 100
                    messagebox.showinfo("完成", "所有文件处理完毕。")
                else:
                    messagebox.showwarning("结束", "处理结束，但存在错误，请查看日志。")

        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    def _start(self):
        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()

        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("错误", "请选择有效的输入文件夹。")
            return
        if not output_dir:
            messagebox.showerror("错误", "请选择输出文件夹。")
            return

        self.start_btn.configure(state='disabled')
        self.progress['value'] = 0
        self._append_log("=" * 70)
        self._append_log("开始处理...")

        recursive = self.recursive_var.get()
        self.worker = threading.Thread(
            target=self._work,
            args=(input_dir, output_dir, recursive),
            daemon=True,
        )
        self.worker.start()

    def _work(self, input_dir, output_dir, recursive):
        ok = True
        try:
            run_process(
                input_dir, output_dir, recursive,
                log=lambda m: self.msg_queue.put(('log', m)),
                progress_cb=lambda v: self.msg_queue.put(('progress', v)),
            )
        except Exception as e:
            ok = False
            self.msg_queue.put(('log', f"[异常] {e}"))
        self.msg_queue.put(('done', ok))


if __name__ == '__main__':
    app = CalibApp()
    app.mainloop()