import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path


class BinToRawConverter:
    """负责按帧长度和头尾剔除转换二进制文件"""

    def __init__(self, log_callback=None, progress_callback=None):
        self.log = log_callback or print
        self.progress = progress_callback
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def convert_file(self, src_path, dst_path, frame_len, skip_head, skip_tail):
        """转换单个文件，返回 (成功标志, 错误信息)"""
        try:
            file_size = os.path.getsize(src_path)
            total_frames, remainder = divmod(file_size, frame_len)
            if remainder != 0:
                self.log(f"警告: {src_path} 末尾 {remainder} 字节不构成完整帧，将被忽略")

            if frame_len <= skip_head + skip_tail:
                raise ValueError(f"帧长度({frame_len})必须大于剔除头尾之和({skip_head}+{skip_tail})")

            bytes_kept_per_frame = frame_len - skip_head - skip_tail
            total_bytes_to_write = total_frames * bytes_kept_per_frame
            written_bytes = 0

            with open(src_path, 'rb') as f_in, open(dst_path, 'wb') as f_out:
                for i in range(total_frames):
                    if self._stop_flag:
                        raise InterruptedError("用户停止转换")

                    frame_data = f_in.read(frame_len)
                    if len(frame_data) < frame_len:
                        break  # 文件意外结束

                    # 切片保留中间有效数据
                    payload = frame_data[skip_head: frame_len - skip_tail]
                    f_out.write(payload)
                    written_bytes += len(payload)

                    if self.progress:
                        self.progress(i + 1, total_frames)

            if written_bytes != total_bytes_to_write:
                self.log(f"警告: 预期写入 {total_bytes_to_write} 字节，实际写入 {written_bytes} 字节")

            self.log(f"成功: {src_path} -> {dst_path} ({total_frames} 帧, 共 {written_bytes} 字节)")
            return True, ""
        except Exception as e:
            return False, str(e)


class Application(tk.Tk):
    """GUI主程序"""

    def __init__(self):
        super().__init__()
        self.title("Bin → Raw 批量转换工具")
        self.geometry("700x550")
        self.resizable(True, True)

        # 转换器实例
        self.converter = BinToRawConverter(log_callback=self.append_log,
                                           progress_callback=self.update_progress)
        self.worker_thread = None

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # 输入路径
        row1 = tk.Frame(self)
        row1.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(row1, text="输入目录:").pack(side=tk.LEFT)
        self.entry_input = tk.Entry(row1)
        self.entry_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tk.Button(row1, text="浏览...", command=self.browse_input).pack(side=tk.LEFT)

        # 输出路径
        row2 = tk.Frame(self)
        row2.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(row2, text="输出目录:").pack(side=tk.LEFT)
        self.entry_output = tk.Entry(row2)
        self.entry_output.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        tk.Button(row2, text="浏览...", command=self.browse_output).pack(side=tk.LEFT)

        # 参数设置区
        param_frame = tk.LabelFrame(self, text="转换参数")
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(param_frame, text="帧长度(字节):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_frame_len = tk.Entry(param_frame, width=10)
        self.entry_frame_len.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.entry_frame_len.insert(0, "1024")  # 默认值

        tk.Label(param_frame, text="剔除头部(字节):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.entry_head = tk.Entry(param_frame, width=10)
        self.entry_head.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        self.entry_head.insert(0, "0")

        tk.Label(param_frame, text="剔除尾部(字节):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.entry_tail = tk.Entry(param_frame, width=10)
        self.entry_tail.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        self.entry_tail.insert(0, "0")

        # 选项
        self.recursive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(param_frame, text="包含子目录", variable=self.recursive_var).grid(row=1, column=0, columnspan=2,
                                                                                         sticky=tk.W, padx=5, pady=2)

        # 控制按钮
        ctrl_frame = tk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_start = tk.Button(ctrl_frame, text="开始转换", command=self.start_conversion, bg="#4CAF50", fg="white",
                                   width=15)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop = tk.Button(ctrl_frame, text="停止", command=self.stop_conversion, bg="#f44336", fg="white",
                                  width=15, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        # 日志区
        self.log_text = scrolledtext.ScrolledText(self, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def browse_input(self):
        path = filedialog.askdirectory(title="选择包含 .bin 文件的目录")
        if path:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)

    def append_log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def update_progress(self, current, total):
        if total > 0:
            percent = int(current / total * 100)
            self.progress['value'] = percent
            self.update_idletasks()

    def validate_params(self):
        try:
            frame_len = int(self.entry_frame_len.get())
            skip_head = int(self.entry_head.get())
            skip_tail = int(self.entry_tail.get())
            if frame_len <= 0 or skip_head < 0 or skip_tail < 0:
                raise ValueError
            if skip_head + skip_tail >= frame_len:
                raise ValueError("剔除头尾之和不能达到或超过帧长度")
            return frame_len, skip_head, skip_tail
        except ValueError as e:
            messagebox.showerror("参数错误", f"请输入有效的正整数，且头尾剔除之和需小于帧长度。\n{e}")
            return None

    def start_conversion(self):
        input_dir = self.entry_input.get().strip()
        output_dir = self.entry_output.get().strip()
        if not input_dir or not output_dir:
            messagebox.showerror("路径错误", "请输入输入和输出目录")
            return

        params = self.validate_params()
        if not params:
            return
        frame_len, skip_head, skip_tail = params

        recursive = self.recursive_var.get()

        # 禁用开始按钮，启用停止按钮
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress['value'] = 0
        self.log_text.delete(1.0, tk.END)

        # 启动工作线程
        self.worker_thread = threading.Thread(
            target=self.convert_all,
            args=(input_dir, output_dir, frame_len, skip_head, skip_tail, recursive),
            daemon=True
        )
        self.converter._stop_flag = False
        self.worker_thread.start()

    def convert_all(self, input_dir, output_dir, frame_len, skip_head, skip_tail, recursive):
        try:
            src_path = Path(input_dir)
            dst_path = Path(output_dir)
            dst_path.mkdir(parents=True, exist_ok=True)

            pattern = "**/*.bin" if recursive else "*.bin"
            bin_files = list(src_path.glob(pattern))

            total_files = len(bin_files)
            self.append_log(f"找到 {total_files} 个 .bin 文件")

            success_count = 0
            for idx, bin_file in enumerate(bin_files, 1):
                if self.converter._stop_flag:
                    self.append_log("转换已被用户停止")
                    break

                rel_path = bin_file.relative_to(src_path)
                out_file = dst_path / rel_path.with_suffix(".raw")
                out_file.parent.mkdir(parents=True, exist_ok=True)

                self.append_log(f"[{idx}/{total_files}] 正在处理: {bin_file}")
                ok, error = self.converter.convert_file(str(bin_file), str(out_file), frame_len, skip_head, skip_tail)
                if ok:
                    success_count += 1
                else:
                    self.append_log(f"错误: {error}")

            self.append_log(f"转换完成: 成功 {success_count}/{total_files}")
        except Exception as e:
            self.append_log(f"发生异常: {e}")
        finally:
            # 恢复按钮状态
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.progress['value'] = 0

    def stop_conversion(self):
        if self.converter:
            self.converter.stop()
        self.btn_stop.config(state=tk.DISABLED)

    def on_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if messagebox.askokcancel("退出", "转换正在进行，确定要退出吗？"):
                self.converter.stop()
                self.destroy()
        else:
            self.destroy()


if __name__ == "__main__":
    app = Application()
    app.mainloop()