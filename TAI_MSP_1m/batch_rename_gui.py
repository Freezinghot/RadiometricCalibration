import os
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, Scrollbar, Button, Label, Frame

class RenameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("批量重命名 .tif → .raw (递归子文件夹)")
        self.root.geometry("600x450")
        self.root.resizable(False, False)

        # 当前选择的目录
        self.dir_path = tk.StringVar()

        # 顶部：目录选择和显示
        top_frame = Frame(root)
        top_frame.pack(pady=10)

        Label(top_frame, text="目标文件夹：").pack(side=tk.LEFT)
        self.dir_label = Label(top_frame, text="未选择", relief=tk.SUNKEN, width=40, anchor="w")
        self.dir_label.pack(side=tk.LEFT, padx=5)
        Button(top_frame, text="浏览...", command=self.select_dir).pack(side=tk.LEFT)

        # 中部：文件列表（显示相对路径）
        list_frame = Frame(root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.listbox = Listbox(list_frame, selectmode=tk.EXTENDED)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        # 底部：操作按钮和状态
        bottom_frame = Frame(root)
        bottom_frame.pack(pady=10)

        Button(bottom_frame, text="刷新列表", command=self.refresh_list).pack(side=tk.LEFT, padx=5)
        Button(bottom_frame, text="开始重命名", command=self.rename_files, bg="lightblue").pack(side=tk.LEFT, padx=5)
        Button(bottom_frame, text="退出", command=root.quit).pack(side=tk.LEFT, padx=5)

        self.status_label = Label(root, text="就绪", relief=tk.SUNKEN, anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def select_dir(self):
        """弹出文件夹选择对话框"""
        dir_selected = filedialog.askdirectory()
        if dir_selected:
            self.dir_path.set(dir_selected)
            self.dir_label.config(text=dir_selected)
            self.refresh_list()

    def refresh_list(self):
        """刷新列表，显示当前目录及所有子目录下的 .tif 文件（显示相对路径）"""
        self.listbox.delete(0, tk.END)
        dir_path = self.dir_path.get()
        if not dir_path or not os.path.isdir(dir_path):
            self.status_label.config(text="请选择有效目录")
            return

        try:
            tif_files = []
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    if f.lower().endswith('.tif'):
                        # 计算相对路径（相对于 dir_path）
                        rel_path = os.path.relpath(os.path.join(root, f), dir_path)
                        tif_files.append(rel_path)

            if not tif_files:
                self.status_label.config(text="未找到 .tif 文件")
            else:
                self.status_label.config(text=f"找到 {len(tif_files)} 个 .tif 文件（含子文件夹）")
            for path in sorted(tif_files):
                self.listbox.insert(tk.END, path)
        except Exception as e:
            messagebox.showerror("错误", f"读取目录失败：{e}")

    def rename_files(self):
        """执行重命名操作（递归处理所有子文件夹）"""
        dir_path = self.dir_path.get()
        if not dir_path or not os.path.isdir(dir_path):
            messagebox.showwarning("警告", "请先选择有效目录")
            return

        # 重新扫描所有 .tif 文件
        try:
            tif_files = []
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    if f.lower().endswith('.tif'):
                        tif_files.append((root, f))
        except Exception as e:
            messagebox.showerror("错误", f"无法读取目录：{e}")
            return

        if not tif_files:
            messagebox.showinfo("提示", "没有 .tif 文件需要重命名")
            return

        renamed_count = 0
        skipped_count = 0
        error_count = 0
        error_details = []

        for root, old_name in tif_files:
            # 使用 os.path.splitext 安全分离扩展名
            base, ext = os.path.splitext(old_name)
            new_name = base + '.raw'

            old_path = os.path.join(root, old_name)
            new_path = os.path.join(root, new_name)

            # 如果新文件已存在，跳过
            if os.path.exists(new_path):
                skipped_count += 1
                continue

            try:
                os.rename(old_path, new_path)
                renamed_count += 1
            except Exception as e:
                error_count += 1
                error_details.append(f"{old_name} (在 {root}) : {e}")

        # 刷新列表显示新状态
        self.refresh_list()

        # 显示结果消息
        msg = f"重命名完成：成功 {renamed_count} 个"
        if skipped_count:
            msg += f"，跳过 {skipped_count} 个（目标文件已存在）"
        if error_count:
            msg += f"，失败 {error_count} 个"
        self.status_label.config(text=msg)

        # 如果有错误，显示详细错误信息（可选）
        if error_details:
            detail_msg = "\n".join(error_details[:10])  # 最多显示前10条
            if len(error_details) > 10:
                detail_msg += "\n... 还有更多错误"
            messagebox.showwarning("部分失败", f"{msg}\n\n错误详情：\n{detail_msg}")
        else:
            messagebox.showinfo("完成", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = RenameApp(root)
    root.mainloop()