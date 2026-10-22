# -*- coding: utf-8 -*-
"""
TIF 波段文件提取工具

功能：
    1. 指定父文件夹和输出文件夹；
    2. 在父文件夹下递归查找名为 "tif" 的文件夹（不区分大小写）；
    3. 在找到的 tif 文件夹中查找 .tif 文件；
    4. 按文件名开头 B1_ / B2_ / B3_ / B4_ / P_ 区分波段，每个波段只复制一个文件；
    5. 从完整路径中提取 CMOS / 增益 / TDI / 亮度 信息，
       与文件名的波段 + 唯一标识拼成新的文件名，例如：
       原始:  F:\\...\\A_CMOS\\B\\增益2\\TDI2\\亮度1\\tif\\
              B1_00000000_000000000048224F_w2272_h1000_pMono12.tif
       输出:  export_dir\\CMOSA_GAIN2_TDI2_BRIGHTNESS1_B1_000000000048224F.tif
    6. GUI 显示处理进度。
"""

import os
import sys
import queue
import shutil
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# 需要提取的波段前缀（每个前缀在同一个 tif 文件夹里只复制一个文件）
PREFIXES = ["B1_", "B2_", "B3_", "B4_", "P_"]

# 目标文件夹名称（不区分大小写）
TIF_DIR_NAME = "tif"

# 关键词表（不区分大小写匹配路径段）
KEYWORDS = {
    "cmos":       ["cmos"],
    "gain":       ["增益", "gain"],
    "tdi":        ["tdi"],
    "brightness": ["亮度", "brightness", "bright"],
}


# ======================================================================
# 路径信息解析
# ======================================================================
def _extract_value(part, keywords):
    """从单个路径段中提取关键词对应的值。
    例如: part='A_CMOS', keywords=['cmos']  ->  'A'
          part='增益2',   keywords=['增益','gain'] -> '2'
          part='TDI2',   keywords=['tdi']      -> '2'
    """
    lower = part.lower()
    for kw in keywords:
        idx = lower.find(kw.lower())
        if idx != -1:
            # 去掉关键词本身，保留其余部分
            result = part[:idx] + part[idx + len(kw):]
            result = result.strip(" _-.")
            if result:
                # 多余的分隔符清理一下（例如 "A__B" -> "A_B"）
                tokens = [t for t in result.split("_") if t]
                if tokens:
                    return "_".join(tokens)
    return ""


def parse_path_info(file_path):
    """从完整文件路径中提取 CMOS、增益、TDI、亮度 四个信息。"""
    parts = file_path.replace("\\", "/").split("/")
    dir_parts = parts[:-1]  # 去掉文件名

    info = {"cmos": "", "gain": "", "tdi": "", "brightness": ""}
    for part in dir_parts:
        for key, kws in KEYWORDS.items():
            if not info[key]:
                v = _extract_value(part, kws)
                if v:
                    info[key] = v
    return info


def parse_filename(filename):
    """解析文件名，返回 (波段, 唯一标识)。

    示例: 'B1_00000000_000000000048224F_w2272_h1000_pMono12.tif'
          -> ('B1', '000000000048224F')
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    band = parts[0] if parts else ""
    if len(parts) >= 3:
        uid = parts[2]
    elif len(parts) >= 2:
        uid = parts[1]
    else:
        uid = ""
    return band, uid


def make_output_name(info, band, uid, fallback=""):
    """拼接新的输出文件名。"""
    tokens = []
    if info["cmos"]:
        tokens.append("CMOS{}".format(info["cmos"]))
    if info["gain"]:
        tokens.append("GAIN{}".format(info["gain"]))
    if info["tdi"]:
        tokens.append("TDI{}".format(info["tdi"]))
    if info["brightness"]:
        tokens.append("BRIGHTNESS{}".format(info["brightness"]))
    if band:
        tokens.append(band)
    if uid:
        tokens.append(uid)

    if not tokens:
        # 兜底：完全没解析出来就用原文件名
        return fallback if fallback else "unknown.tif"
    return "_".join(tokens) + ".tif"


# ======================================================================
# 工具函数
# ======================================================================
def unique_path(path):
    """若目标路径已存在，则自动追加 _1、_2 …，避免覆盖已有文件。"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        cand = "{}_{}{}".format(base, i, ext)
        if not os.path.exists(cand):
            return cand
        i += 1


def open_folder(path):
    """用系统默认方式打开文件夹。"""
    if not path or not os.path.isdir(path):
        messagebox.showwarning("提示", "输出文件夹不存在。")
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("错误", "无法打开文件夹：{}".format(e))


# ======================================================================
# GUI
# ======================================================================
class TifExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TIF 波段文件提取工具（路径信息重命名）")
        self.root.geometry("900x640")
        self.root.minsize(760, 540)

        self.msg_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None

        self._build_ui()
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(6, weight=1)

        # 父文件夹
        ttk.Label(main, text="父文件夹：").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.parent_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.parent_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(main, text="浏览…", width=10, command=self._choose_parent).grid(row=0, column=2, padx=4, pady=4)

        # 输出文件夹
        ttk.Label(main, text="输出文件夹：").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.out_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(main, text="浏览…", width=10, command=self._choose_out).grid(row=1, column=2, padx=4, pady=4)

        # 说明
        tip = ("新文件名格式示例：\n"
               "    CMOSA_GAIN2_TDI2_BRIGHTNESS1_B1_000000000048224F.tif\n"
               "（从路径中提取 CMOS/增益/TDI/亮度，从原文件名提取波段与唯一标识）")
        ttk.Label(main, text=tip, foreground="#555555", justify="left").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))

        # 选项
        self.keep_structure = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            main,
            text="在输出文件夹中保留原有的相对目录结构（默认不勾选，全部平铺存放；重名自动加序号）",
            variable=self.keep_structure
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=2)

        # 按钮
        btns = ttk.Frame(main)
        btns.grid(row=4, column=0, columnspan=3, sticky="ew", padx=4, pady=6)
        self.start_btn = ttk.Button(btns, text="开始处理", width=14, command=self.start)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ttk.Button(btns, text="停止", width=10, command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="打开输出文件夹", width=16,
                   command=lambda: open_folder(self.out_var.get().strip())).pack(side="left")
        ttk.Button(btns, text="清空日志", width=10, command=self._clear_log).pack(side="right")

        # 进度条
        self.progress = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", padx=4, pady=(4, 2))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main, textvariable=self.status_var, anchor="w").grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 4))

        # 日志区
        log_frame = ttk.LabelFrame(main, text="处理日志", padding=4)
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main.rowconfigure(7, weight=1)

        self.log_text = tk.Text(log_frame, wrap="none", height=12, state="disabled",
                                font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.log_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # ------------------------------------------------------------------
    def _choose_parent(self):
        d = filedialog.askdirectory(title="选择父文件夹")
        if d:
            self.parent_var.set(os.path.normpath(d))

    def _choose_out(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.out_var.set(os.path.normpath(d))

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 4000:
            self.log_text.delete("1.0", "2500.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    def start(self):
        if self.worker and self.worker.is_alive():
            return

        parent = self.parent_var.get().strip()
        out = self.out_var.get().strip()

        if not parent or not os.path.isdir(parent):
            messagebox.showerror("错误", "请选择有效的父文件夹！")
            return
        if not out:
            messagebox.showerror("错误", "请选择输出文件夹！")
            return
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            messagebox.showerror("错误", "无法创建输出文件夹：\n{}".format(e))
            return

        self.stop_flag.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(mode="determinate", value=0, maximum=100)
        self.status_var.set("开始处理…")

        self.worker = threading.Thread(
            target=self._run,
            args=(parent, out, self.keep_structure.get()),
            daemon=True
        )
        self.worker.start()

    def stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_flag.set()
            self.status_var.set("正在停止…")
            self.stop_btn.configure(state="disabled")

    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]

                if kind == "log":
                    self._append_log(msg[1])

                elif kind == "scan_start":
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(12)
                    self.status_var.set("正在扫描 tif 文件夹…")

                elif kind == "scan_done":
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0,
                                            maximum=max(msg[1], 1))
                    self.status_var.set("共找到 {} 个 tif 文件夹".format(msg[1]))

                elif kind == "progress":
                    _, i, total, d, n = msg
                    self.progress.configure(maximum=max(total, 1), value=i)
                    self.status_var.set(
                        "正在处理 {}/{}：{} （本目录复制 {} 个）".format(i, total, d, n))

                elif kind == "done":
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status_var.set("已取消" if msg[1] else "处理完成")
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # 子线程：处理逻辑
    # ------------------------------------------------------------------
    def _run(self, parent, out, keep_structure):
        q = self.msg_queue
        cancelled = False
        try:
            q.put(("scan_start",))
            q.put(("log", "开始扫描：{}".format(parent)))

            tif_dirs = self._scan(parent)
            if self.stop_flag.is_set():
                q.put(("scan_done", 0))
                q.put(("log", "已取消。"))
                q.put(("done", True))
                return

            q.put(("scan_done", len(tif_dirs)))
            q.put(("log", "共找到 {} 个名为 tif 的文件夹。".format(len(tif_dirs))))

            if not tif_dirs:
                q.put(("log", "未找到任何 tif 文件夹，程序结束。"))
                q.put(("done", False))
                return

            total_copied = 0
            stat = {p: 0 for p in PREFIXES}

            for i, d in enumerate(tif_dirs, 1):
                if self.stop_flag.is_set():
                    cancelled = True
                    q.put(("log", "已取消。"))
                    break

                n, s = self._process_dir(d, parent, out, keep_structure)
                total_copied += n
                for k, v in s.items():
                    stat[k] += v
                q.put(("progress", i, len(tif_dirs), d, n))

            if not cancelled:
                q.put(("log", "-" * 70))
                q.put(("log", "全部完成！共复制 {} 个文件。".format(total_copied)))
                for p in PREFIXES:
                    q.put(("log", "    {:<5} : {} 个".format(p, stat[p])))

        except Exception as e:
            q.put(("log", "发生错误：{}".format(e)))
        finally:
            q.put(("done", cancelled))

    # ------------------------------------------------------------------
    def _scan(self, parent):
        """递归查找所有名为 tif 的文件夹（不区分大小写）。"""
        tif_dirs = []

        if os.path.basename(os.path.abspath(parent)).lower() == TIF_DIR_NAME:
            tif_dirs.append(os.path.abspath(parent))

        for dirpath, dirnames, filenames in os.walk(parent):
            if self.stop_flag.is_set():
                break
            matched = [d for d in dirnames if d.lower() == TIF_DIR_NAME]
            for d in matched:
                tif_dirs.append(os.path.join(dirpath, d))
            # 不再深入 tif 文件夹内部继续查找
            dirnames[:] = [d for d in dirnames if d.lower() != TIF_DIR_NAME]

        return tif_dirs

    # ------------------------------------------------------------------
    def _process_dir(self, tif_dir, parent, out, keep_structure):
        """处理单个 tif 文件夹，返回 (复制数量, 各波段统计)。"""
        q = self.msg_queue
        stat = {p: 0 for p in PREFIXES}

        try:
            entries = os.listdir(tif_dir)
        except OSError as e:
            q.put(("log", "无法读取目录 {}：{}".format(tif_dir, e)))
            return 0, stat

        files = [
            f for f in entries
            if f.lower().endswith(".tif") and os.path.isfile(os.path.join(tif_dir, f))
        ]
        files.sort(key=lambda s: s.lower())

        # 按前缀分组，每个前缀只取第一个
        chosen = {}
        for f in files:
            upper = f.upper()
            for p in PREFIXES:
                if upper.startswith(p):
                    if p not in chosen:
                        chosen[p] = f
                    break

        if not chosen:
            q.put(("log", "[跳过] {} （没有匹配的 tif 文件）".format(tif_dir)))
            return 0, stat

        # 目标目录
        if keep_structure:
            rel = os.path.relpath(tif_dir, parent)
            dest_dir = os.path.join(out, rel)
        else:
            dest_dir = out
        os.makedirs(dest_dir, exist_ok=True)

        count = 0
        for p in PREFIXES:
            if p not in chosen:
                continue

            src = os.path.join(tif_dir, chosen[p])
            # 从完整路径提取 CMOS / 增益 / TDI / 亮度
            info = parse_path_info(src)
            # 从文件名提取波段与唯一标识
            band, uid = parse_filename(chosen[p])
            # 生成新文件名
            new_name = make_output_name(info, band, uid, fallback=chosen[p])
            dst = unique_path(os.path.join(dest_dir, new_name))

            try:
                shutil.copy2(src, dst)
                count += 1
                stat[p] += 1
                q.put(("log", "[{}] {}  ->  {}".format(
                    p.rstrip("_"), src, os.path.basename(dst))))
            except Exception as e:
                q.put(("log", "复制失败 {}：{}".format(src, e)))

        return count, stat


# ======================================================================
def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    TifExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()