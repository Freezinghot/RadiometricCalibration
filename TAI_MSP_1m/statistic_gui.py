# -*- coding: utf-8 -*-
# @File  : statistic_gui.py
# @Author: Freezinghot
# @Date  : 2026/7/24
# @Desc  :
# -*- coding: utf-8 -*-
# @File  : statistic_images_gui.py
# @Author: Freezinghot
# @Date  : 2026/7/24
# @Desc  : TIF图像列方向均值统计工具 - GUI版本

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image
import os
import csv
from pathlib import Path
from typing import List, Union
import threading


class TIFStatisticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TIF图像列方向均值统计工具")
        self.root.geometry("600x400")

        # 变量
        self.input_dir = tk.StringVar()
        self.output_file = tk.StringVar()
        self.pattern = tk.StringVar(value="B1*.tif")
        self.recursive = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="就绪")

        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 输入目录
        ttk.Label(main_frame, text="输入目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2)

        # 匹配模式
        ttk.Label(main_frame, text="匹配模式:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.pattern, width=30).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(main_frame, text="(例如: B1*.tif)").grid(row=1, column=2, sticky=tk.W)

        # 递归搜索
        ttk.Checkbutton(main_frame, text="递归搜索子目录", variable=self.recursive).grid(row=2, column=1, sticky=tk.W, pady=5)

        # 输出文件
        ttk.Label(main_frame, text="输出文件:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_file, width=50).grid(row=3, column=1, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_output).grid(row=3, column=2)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # 状态标签
        ttk.Label(main_frame, textvariable=self.status).grid(row=5, column=0, columnspan=3, pady=5)

        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="开始处理", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # 日志区域
        ttk.Label(main_frame, text="日志:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.log_text = tk.Text(main_frame, height=10, width=70)
        self.log_text.grid(row=8, column=0, columnspan=3, pady=5)

        # 滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=8, column=3, sticky=(tk.N, tk.S))
        self.log_text.config(yscrollcommand=scrollbar.set)

    def browse_input(self):
        """浏览输入目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
            self.log(f"选择输入目录: {directory}")

    def browse_output(self):
        """浏览输出文件"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.output_file.set(file_path)
            self.log(f"选择输出文件: {file_path}")

    def log(self, message):
        """添加日志"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_processing(self):
        """开始处理"""
        # 验证输入
        if not self.input_dir.get():
            messagebox.showerror("错误", "请选择输入目录")
            return

        if not self.output_file.get():
            messagebox.showerror("错误", "请指定输出文件")
            return

        # 禁用按钮，启动进度条
        self.progress.start()
        self.status.set("正在处理...")

        # 在新线程中处理
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()

    def process_images(self):
        """处理图像"""
        try:
            # 获取文件列表
            input_dir = Path(self.input_dir.get())
            pattern = self.pattern.get()
            recursive = self.recursive.get()

            self.log(f"开始搜索文件...")
            self.log(f"目录: {input_dir}")
            self.log(f"模式: {pattern}")

            if recursive:
                tif_files = list(input_dir.rglob(pattern))
            else:
                tif_files = list(input_dir.glob(pattern))

            self.log(f"找到 {len(tif_files)} 个匹配文件")

            if not tif_files:
                messagebox.showwarning("警告", "没有找到匹配的TIF文件")
                self.status.set("未找到文件")
                self.progress.stop()
                return

            # 处理图像
            self.log("开始处理图像...")
            means = self.calculate_column_means(tif_files)

            # 保存结果
            self.log("保存结果...")
            self.save_means_to_csv(means, self.output_file.get())

            # 完成
            self.status.set("处理完成")
            self.progress.stop()

            messagebox.showinfo("完成",
                                f"处理完成！\n"
                                f"处理的图像数量: {len(tif_files)}\n"
                                f"输出文件: {self.output_file.get()}"
                                )

        except Exception as e:
            self.status.set("处理出错")
            self.progress.stop()
            self.log(f"错误: {str(e)}")
            messagebox.showerror("错误", f"处理过程中出错:\n{str(e)}")

    def calculate_column_means(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """计算列方向均值"""
        if not image_paths:
            raise ValueError("图像列表为空")

        all_images = []
        total_rows = 0
        image_width = None

        self.log(f"正在读取 {len(image_paths)} 个图像文件...")

        for idx, img_path in enumerate(image_paths, 1):
            try:
                img = Image.open(img_path)
                img_array = np.array(img, dtype=np.float64)

                if len(img_array.shape) == 3:
                    img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

                if image_width is None:
                    image_width = img_array.shape[1]
                elif img_array.shape[1] != image_width:
                    raise ValueError(f"图像 {img_path} 的列数不一致")

                all_images.append(img_array)
                total_rows += img_array.shape[0]

                if idx % 10 == 0:
                    self.log(f"  已读取 {idx}/{len(image_paths)} 个图像")

            except Exception as e:
                self.log(f"读取图像 {img_path} 时出错: {e}")
                continue

        if not all_images:
            raise ValueError("没有成功读取任何图像")

        self.log(f"总行数: {total_rows}, 列数: {image_width}")
        self.log("正在计算列方向均值...")

        column_sum = np.zeros(image_width, dtype=np.float64)
        total_rows_actual = 0

        for img_array in all_images:
            column_sum += np.sum(img_array, axis=0)
            total_rows_actual += img_array.shape[0]

        column_means = column_sum / total_rows_actual

        self.log(f"计算完成，均值数组长度: {len(column_means)}")

        return column_means

    def save_means_to_csv(self, means: np.ndarray, output_path: Union[str, Path]):
        """保存为CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [['Column_Index', 'Mean_Value']]
        for idx, value in enumerate(means):
            data.append([idx, value])

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

        self.log(f"CSV文件已保存: {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TIFStatisticApp(root)
    root.mainloop()