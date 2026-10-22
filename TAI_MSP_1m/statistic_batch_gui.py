# -*- coding: utf-8 -*-
# @File  : statistic_images_gui.py
# @Author: Freezinghot
# @Date  : 2026/7/24
# @Desc  : TIF图像列方向均值统计工具 - 手动+自动批处理双模式（文件名自动含模式标识）

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
        self.root.geometry("650x450")

        # 变量
        self.input_dir = tk.StringVar()
        self.output_file = tk.StringVar()
        self.pattern = tk.StringVar(value="B1*.tif")
        self.recursive = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="就绪")

        self.setup_ui()

    def setup_ui(self):
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

        # 递归/批处理复选框
        self.recursive_check = ttk.Checkbutton(
            main_frame,
            text="递归搜索子目录并自动批处理（为每个tif文件夹生成结果）",
            variable=self.recursive,
            command=self.toggle_output_mode
        )
        self.recursive_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

        # 输出文件（仅手动模式显示）
        self.output_label = ttk.Label(main_frame, text="输出文件:")
        self.output_entry = ttk.Entry(main_frame, textvariable=self.output_file, width=50)
        self.output_button = ttk.Button(main_frame, text="浏览...", command=self.browse_output)

        self.output_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_entry.grid(row=3, column=1, padx=5)
        self.output_button.grid(row=3, column=2)

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
        self.log_text = tk.Text(main_frame, height=10, width=75)
        self.log_text.grid(row=8, column=0, columnspan=3, pady=5)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=8, column=3, sticky=(tk.N, tk.S))
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.toggle_output_mode()

    def toggle_output_mode(self):
        if self.recursive.get():
            self.output_label.grid_remove()
            self.output_entry.grid_remove()
            self.output_button.grid_remove()
        else:
            self.output_label.grid()
            self.output_entry.grid()
            self.output_button.grid()

    def browse_input(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
            self.log(f"选择输入目录: {directory}")

    def browse_output(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.output_file.set(file_path)
            self.log(f"选择输出文件: {file_path}")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_processing(self):
        if not self.input_dir.get():
            messagebox.showerror("错误", "请选择输入目录")
            return
        if not self.recursive.get() and not self.output_file.get():
            messagebox.showerror("错误", "请指定输出文件")
            return

        self.progress.start()
        self.status.set("正在处理...")

        thread = threading.Thread(target=self.run_processing)
        thread.daemon = True
        thread.start()

    def run_processing(self):
        try:
            input_dir = Path(self.input_dir.get())
            pattern = self.pattern.get()

            if self.recursive.get():
                # ========== 自动批处理模式 ==========
                self.log("开始递归搜索所有名为 'tif' 的文件夹...")
                tif_folders = [p for p in input_dir.rglob("tif") if p.is_dir()]
                self.log(f"找到 {len(tif_folders)} 个 'tif' 文件夹")

                if not tif_folders:
                    messagebox.showwarning("警告", "未找到任何名为 'tif' 的文件夹")
                    self.status.set("未找到 tif 文件夹")
                    self.progress.stop()
                    return

                # 提取模式后缀（去除扩展名和 * 号）
                base_pattern = os.path.splitext(pattern)[0]   # e.g., "B1*"
                pattern_suffix = base_pattern.replace('*', '')  # e.g., "B1"

                success_count = 0
                for idx, tif_folder in enumerate(tif_folders, 1):
                    self.log(f"[{idx}/{len(tif_folders)}] 处理文件夹: {tif_folder}")
                    try:
                        image_files = sorted(tif_folder.glob(pattern))
                        self.log(f"  找到 {len(image_files)} 个匹配图像")
                        if not image_files:
                            self.log("  没有匹配的图像，跳过该文件夹")
                            continue

                        means = self.calculate_column_means(image_files)

                        # 生成输出文件名：路径 + 模式后缀
                        try:
                            relative = tif_folder.relative_to(input_dir)
                        except ValueError:
                            relative = Path("")
                        parts = [input_dir.name] + list(relative.parent.parts) if relative.parent.parts else [input_dir.name]
                        if pattern_suffix:
                            output_name = "_".join(parts) + "_" + pattern_suffix + ".csv"
                        else:
                            output_name = "_".join(parts) + ".csv"
                        output_path = tif_folder / output_name

                        self.save_means_to_csv(means, output_path)
                        self.log(f"  结果已保存至: {output_path.name}")
                        success_count += 1

                    except Exception as e:
                        self.log(f"  处理文件夹时出错: {e}")
                        continue

                self.status.set(f"批处理完成，成功处理 {success_count}/{len(tif_folders)} 个文件夹")
                self.progress.stop()
                messagebox.showinfo("完成", f"批处理完成！成功处理 {success_count} 个 tif 文件夹。")

            else:
                # ========== 手动模式 ==========
                self.log("开始搜索文件...")
                tif_files = sorted(input_dir.glob(pattern))
                self.log(f"找到 {len(tif_files)} 个匹配文件")

                if not tif_files:
                    messagebox.showwarning("警告", "没有找到匹配的TIF文件")
                    self.status.set("未找到文件")
                    self.progress.stop()
                    return

                means = self.calculate_column_means(tif_files)
                self.save_means_to_csv(means, self.output_file.get())

                self.status.set("处理完成")
                self.progress.stop()
                messagebox.showinfo("完成",
                                    f"处理完成！\n"
                                    f"处理的图像数量: {len(tif_files)}\n"
                                    f"输出文件: {self.output_file.get()}")

        except Exception as e:
            self.status.set("处理出错")
            self.progress.stop()
            self.log(f"错误: {str(e)}")
            messagebox.showerror("错误", f"处理过程中出错:\n{str(e)}")

    def calculate_column_means(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        if not image_paths:
            raise ValueError("图像列表为空")

        all_images = []
        total_rows = 0
        image_width = None

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
                self.log(f"  读取图像 {img_path} 时出错: {e}")
                continue

        if not all_images:
            raise ValueError("没有成功读取任何图像")

        column_sum = np.zeros(image_width, dtype=np.float64)
        total_rows_actual = 0
        for img_array in all_images:
            column_sum += np.sum(img_array, axis=0)
            total_rows_actual += img_array.shape[0]

        return column_sum / total_rows_actual

    def save_means_to_csv(self, means: np.ndarray, output_path: Union[str, Path]):
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