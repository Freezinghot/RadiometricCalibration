# --*-- conding:utf-8 --*--
# @File  : 2m_spectral_statistic.py
# @Author: FH
# @Date  : 2026/9/9
# @Desc  :
import os
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

class RawImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Raw 图像均值统计工具")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # 变量
        self.root_dir = tk.StringVar()
        self.center_row = tk.StringVar(value="5280")   # 默认中心行（0-based）
        self.center_col = tk.StringVar(value="3270")   # 默认中心列（0-based）
        self.window_size = tk.StringVar(value="11")     # 窗口边长，奇数
        self.output_path = tk.StringVar()
        self.extension = tk.StringVar(value=".raw")    # raw 文件扩展名

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 根目录选择
        tk.Label(self.root, text="根目录:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.root_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="浏览...", command=self.browse_root).grid(row=0, column=2, padx=5, pady=5)

        # 中心行坐标
        tk.Label(self.root, text="中心行坐标 (0-based):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.center_row, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # 中心列坐标
        tk.Label(self.root, text="中心列坐标 (0-based):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.center_col, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # 窗口大小
        tk.Label(self.root, text="窗口边长 (奇数):").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.window_size, width=10).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # raw 文件扩展名
        tk.Label(self.root, text="Raw 文件扩展名:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.extension, width=10).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # 输出 CSV 路径
        tk.Label(self.root, text="输出 CSV 文件:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.output_path, width=50).grid(row=5, column=1, padx=5, pady=5)
        tk.Button(self.root, text="保存为...", command=self.browse_output).grid(row=5, column=2, padx=5, pady=5)

        # 运行按钮
        self.run_button = tk.Button(self.root, text="开始分析", command=self.start_analysis)
        self.run_button.grid(row=6, column=1, pady=20)

        # 状态显示
        self.status_label = tk.Label(self.root, text="就绪", fg="blue")
        self.status_label.grid(row=7, column=0, columnspan=3, pady=5)

        # 进度条
        self.progress = ttk.Progressbar(self.root, length=500, mode='determinate')
        self.progress.grid(row=8, column=0, columnspan=3, pady=10)

    def browse_root(self):
        directory = filedialog.askdirectory()
        if directory:
            self.root_dir.set(directory)

    def browse_output(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV 文件", "*.csv")])
        if file_path:
            self.output_path.set(file_path)

    def start_analysis(self):
        # 获取参数
        root_dir = self.root_dir.get().strip()
        if not root_dir:
            messagebox.showerror("错误", "请选择根目录")
            return

        try:
            center_row = int(self.center_row.get())
            center_col = int(self.center_col.get())
            window_size = int(self.window_size.get())
        except ValueError:
            messagebox.showerror("错误", "中心坐标和窗口大小必须为整数")
            return

        if window_size <= 0 or window_size % 2 == 0:
            messagebox.showerror("错误", "窗口大小必须为正奇数")
            return

        output_path = self.output_path.get().strip()
        if not output_path:
            messagebox.showerror("错误", "请指定输出 CSV 文件路径")
            return

        extension = self.extension.get().strip()
        if not extension.startswith("."):
            extension = "." + extension

        # 禁用按钮，启动线程
        self.run_button.config(state=tk.DISABLED)
        self.status_label.config(text="正在分析...", fg="orange")
        self.progress["value"] = 0

        thread = threading.Thread(target=self.analyze,
                                  args=(root_dir, center_row, center_col, window_size, output_path, extension))
        thread.daemon = True
        thread.start()

    def analyze(self, root_dir, center_row, center_col, window_size, output_path, extension):
        try:
            # 获取所有子文件夹
            subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            if not subdirs:
                raise ValueError("根目录下没有子文件夹")

            total_dirs = len(subdirs)
            results = []

            for idx, subdir in enumerate(subdirs):
                subdir_path = os.path.join(root_dir, subdir)
                # 获取该文件夹内所有 raw 文件
                raw_files = [f for f in os.listdir(subdir_path)
                             if f.lower().endswith(extension.lower()) and os.path.isfile(os.path.join(subdir_path, f))]
                if not raw_files:
                    print(f"警告: 文件夹 {subdir} 中没有找到扩展名为 {extension} 的文件，跳过")
                    continue

                folder_means = []
                for raw_file in raw_files:
                    raw_path = os.path.join(subdir_path, raw_file)
                    try:
                        # 读取 raw 文件
                        img = np.fromfile(raw_path, dtype=np.uint8)
                        expected_size = 9520 * 7000
                        if img.size != expected_size:
                            raise ValueError(f"文件大小错误: {raw_path}, 期望 {expected_size} 字节, 实际 {img.size} 字节")
                        img = img.reshape((7000, 9520))

                        # 计算窗口边界
                        half = window_size // 2
                        row_start = max(0, center_row - half)
                        row_end = min(7000, center_row + half + 1)
                        col_start = max(0, center_col - half)
                        col_end = min(9520, center_col + half + 1)

                        # 提取窗口并计算均值
                        window = img[col_start:col_end, row_start:row_end]
                        mean_val = np.mean(window)
                        folder_means.append(mean_val)
                    except Exception as e:
                        print(f"处理文件 {raw_path} 时出错: {e}")
                        continue

                if folder_means:
                    folder_avg = np.mean(folder_means)
                    results.append((subdir, folder_avg))
                else:
                    print(f"警告: 文件夹 {subdir} 中没有成功处理的文件")

                # 更新进度条
                progress_percent = (idx + 1) / total_dirs * 100
                self.root.after(0, self.update_progress, progress_percent, f"已处理 {idx+1}/{total_dirs} 个文件夹")

            # 写入 CSV
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['文件夹名', '图像均值'])
                for subdir, avg in results:
                    writer.writerow([subdir, f"{avg:.6f}"])

            self.root.after(0, self.analysis_complete, f"分析完成，共处理 {len(results)} 个文件夹，结果已保存至 {output_path}")

        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))

    def update_progress(self, value, text):
        self.progress["value"] = value
        self.status_label.config(text=text)

    def analysis_complete(self, message):
        self.status_label.config(text=message, fg="green")
        self.progress["value"] = 100
        self.run_button.config(state=tk.NORMAL)
        messagebox.showinfo("完成", message)

    def analysis_error(self, error_msg):
        self.status_label.config(text=f"错误: {error_msg}", fg="red")
        self.run_button.config(state=tk.NORMAL)
        messagebox.showerror("错误", error_msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = RawImageAnalyzerGUI(root)
    root.mainloop()