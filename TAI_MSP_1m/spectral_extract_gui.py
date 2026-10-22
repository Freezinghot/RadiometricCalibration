import os
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 设置 Matplotlib 后端为 TkAgg，以便嵌入 Tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpectralDataExtractor:
    def __init__(self, root):
        self.root = root
        root.title("光谱定标数据提取器")
        root.geometry("800x700")

        # 数据缓存 (波长, 均值)
        self.data = []

        # ---------- 控制区域 ----------
        top_frame = tk.Frame(root)
        top_frame.pack(pady=5, fill=tk.X, padx=10)

        # 文件夹选择
        tk.Label(top_frame, text="数据文件夹:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar()
        entry_dir = tk.Entry(top_frame, textvariable=self.dir_var, width=30)
        entry_dir.pack(side=tk.LEFT, padx=5)
        btn_browse = tk.Button(top_frame, text="浏览", command=self.browse_folder)
        btn_browse.pack(side=tk.LEFT, padx=5)

        # 列序号
        tk.Label(top_frame, text="列序号:").pack(side=tk.LEFT, padx=(20,5))
        self.col_var = tk.StringVar()
        entry_col = tk.Entry(top_frame, textvariable=self.col_var, width=8)
        entry_col.pack(side=tk.LEFT, padx=5)

        # 按钮
        btn_extract = tk.Button(top_frame, text="提取", command=self.extract_data, width=8)
        btn_extract.pack(side=tk.LEFT, padx=5)
        btn_export = tk.Button(top_frame, text="导出 CSV", command=self.export_csv, width=8)
        btn_export.pack(side=tk.LEFT, padx=5)
        btn_plot = tk.Button(top_frame, text="绘制曲线", command=self.plot_curve, width=8)
        btn_plot.pack(side=tk.LEFT, padx=5)

        # ---------- 表格（结果显示） ----------
        tree_frame = tk.Frame(root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(tree_frame, columns=("wavelength", "mean_value"), show="headings")
        self.tree.heading("wavelength", text="波长 (nm)")
        self.tree.heading("mean_value", text="均值 (Mean_Value)")
        self.tree.column("wavelength", width=150, anchor="center")
        self.tree.column("mean_value", width=300, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # ---------- 图像显示区域 ----------
        plot_frame = tk.LabelFrame(root, text="光谱曲线", padx=5, pady=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 创建 Matplotlib 图形并嵌入
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始清空图形
        self.ax.clear()
        self.ax.set_xlabel("波长 (nm)")
        self.ax.set_ylabel("均值")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

        # ---------- 状态栏 ----------
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dir_var.set(folder)

    def extract_data(self):
        # 清空表格和缓存
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.data.clear()

        # 清空图像
        self.ax.clear()
        self.ax.set_xlabel("波长 (nm)")
        self.ax.set_ylabel("均值")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

        folder = self.dir_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("错误", "请选择一个有效的文件夹")
            return

        try:
            target_col = int(self.col_var.get().strip())
        except ValueError:
            messagebox.showerror("错误", "列序号必须为整数")
            return

        # 获取所有 CSV 文件
        csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
        if not csv_files:
            self.status_var.set("未找到任何 CSV 文件")
            return

        results = []
        for filename in csv_files:
            match = re.search(r'\d+', filename)
            if not match:
                continue
            wavelength = match.group()

            filepath = os.path.join(folder, filename)
            mean_val = None
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if 'Column_Index' not in reader.fieldnames or 'Mean_Value' not in reader.fieldnames:
                        mean_val = "表头缺失"
                    else:
                        found = False
                        for row in reader:
                            try:
                                col_idx = int(row['Column_Index'])
                                if col_idx == target_col:
                                    mean_val = row['Mean_Value'].strip()
                                    found = True
                                    break
                            except (ValueError, KeyError):
                                continue
                        if not found:
                            mean_val = "未找到"
            except Exception as e:
                mean_val = f"读取错误: {e}"

            results.append((wavelength, mean_val))

        if not results:
            self.status_var.set("没有找到包含数字文件名的 CSV 文件")
            return

        # 按波长排序
        try:
            results.sort(key=lambda x: int(x[0]))
        except ValueError:
            results.sort(key=lambda x: x[0])

        self.data = results[:]
        for wave, val in results:
            self.tree.insert("", tk.END, values=(wave, val))

        self.status_var.set(f"完成，共处理 {len(results)} 个文件")

    def export_csv(self):
        if not self.data:
            messagebox.showwarning("提示", "请先执行提取操作，获取数据")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            title="保存光谱数据为 CSV"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["波长 (nm)", "均值 (Mean_Value)"])
                for wave, val in self.data:
                    writer.writerow([wave, val])
            self.status_var.set(f"数据已导出至: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", f"文件已保存至:\n{file_path}")
        except Exception as e:
            messagebox.showerror("导出错误", f"保存失败:\n{e}")

    def plot_curve(self):
        if not self.data:
            messagebox.showwarning("提示", "请先执行提取操作，获取数据")
            return

        # 解析数据
        wavelengths = []
        mean_values = []
        for wave_str, mean_str in self.data:
            try:
                w = float(wave_str)
                m = float(mean_str)
                wavelengths.append(w)
                mean_values.append(m)
            except ValueError:
                continue

        if len(wavelengths) == 0:
            messagebox.showwarning("提示", "没有可用于绘图的有效数值数据")
            return

        # 清空并绘制新图
        self.ax.clear()
        self.ax.plot(wavelengths, mean_values, marker='o', linestyle='-', color='b', linewidth=1, markersize=2)
        self.ax.set_xlabel("Wavelength(nm)")
        self.ax.set_ylabel("Mean Value")
        self.ax.set_title("Spectral Curve")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        # 自动调整坐标轴范围（留边距）
        if len(wavelengths) > 1:
            xmin, xmax = min(wavelengths), max(wavelengths)
            xrange = xmax - xmin
            if xrange == 0:
                xrange = 1
            self.ax.set_xlim(xmin - 0.05*xrange, xmax + 0.05*xrange)
        else:
            self.ax.set_xlim(wavelengths[0]-1, wavelengths[0]+1)

        # 更新画布
        self.canvas.draw()
        self.status_var.set("光谱曲线已更新")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralDataExtractor(root)
    root.mainloop()