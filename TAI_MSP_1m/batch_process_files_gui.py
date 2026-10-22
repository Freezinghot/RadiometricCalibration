import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import shutil
import threading


class CleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文件/文件夹批处理工具")
        self.root.geometry("750x650")
        self.root.resizable(True, True)

        # ---------- 顶部：目录选择 ----------
        frame_top = tk.Frame(root)
        frame_top.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frame_top, text="根目录:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar()
        self.entry_dir = tk.Entry(frame_top, textvariable=self.dir_var, width=50)
        self.entry_dir.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        tk.Button(frame_top, text="浏览...", command=self.browse_dir).pack(side=tk.LEFT)

        # ---------- 中部：筛选条件 ----------
        frame_mid = tk.Frame(root)
        frame_mid.pack(fill=tk.X, padx=10, pady=5)

        # 文件类型
        tk.Label(frame_mid, text="匹配文件后缀(空格分隔):").grid(row=0, column=0, sticky="w", pady=2)
        self.ext_var = tk.StringVar(value=".tif .bin")
        self.entry_ext = tk.Entry(frame_mid, textvariable=self.ext_var, width=40)
        self.entry_ext.grid(row=0, column=1, sticky="ew", padx=5)

        # 文件夹名称
        tk.Label(frame_mid, text="匹配文件夹名(空格分隔):").grid(row=1, column=0, sticky="w", pady=2)
        self.folder_var = tk.StringVar(value="__pycache__")
        self.entry_folder = tk.Entry(frame_mid, textvariable=self.folder_var, width=40)
        self.entry_folder.grid(row=1, column=1, sticky="ew", padx=5)

        frame_mid.columnconfigure(1, weight=1)

        # ---------- 按钮区 ----------
        frame_btn = tk.Frame(root)
        frame_btn.pack(fill=tk.X, padx=10, pady=5)

        self.btn_scan = tk.Button(frame_btn, text="扫描匹配项", command=self.start_scan)
        self.btn_scan.pack(side=tk.LEFT, padx=3)

        self.btn_delete = tk.Button(frame_btn, text="删除匹配项", command=self.start_delete, state=tk.DISABLED)
        self.btn_delete.pack(side=tk.LEFT, padx=3)

        # 新增：移动与复制按钮（初始不可用）
        self.btn_move = tk.Button(frame_btn, text="移动文件到...", command=lambda: self.start_move_copy("move"),
                                  state=tk.DISABLED)
        self.btn_move.pack(side=tk.LEFT, padx=3)

        self.btn_copy = tk.Button(frame_btn, text="复制文件到...", command=lambda: self.start_move_copy("copy"),
                                  state=tk.DISABLED)
        self.btn_copy.pack(side=tk.LEFT, padx=3)

        self.btn_clear = tk.Button(frame_btn, text="清空日志", command=self.clear_log)
        self.btn_clear.pack(side=tk.RIGHT, padx=3)

        # ---------- 日志区域 ----------
        frame_log = tk.Frame(root)
        frame_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(frame_log, wrap=tk.WORD, width=80, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 存储扫描结果
        self.files_to_delete = []
        self.dirs_to_delete = []

    def browse_dir(self):
        directory = filedialog.askdirectory(title="选择要清理的根目录")
        if directory:
            self.dir_var.set(directory)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    # ------------------ 扫描逻辑 ------------------
    def start_scan(self):
        root_dir = self.dir_var.get().strip()
        if not root_dir:
            messagebox.showwarning("警告", "请先选择根目录")
            return
        if not os.path.isdir(root_dir):
            messagebox.showerror("错误", "指定的目录不存在")
            return

        exts = [e.strip().lower() for e in self.ext_var.get().split() if e.strip()]
        if exts:
            exts = [e if e.startswith(".") else f".{e}" for e in exts]

        folders = [f.strip() for f in self.folder_var.get().split() if f.strip()]

        if not exts and not folders:
            messagebox.showwarning("警告", "至少输入一种文件后缀或文件夹名称")
            return

        self.btn_scan.config(state=tk.DISABLED)
        self.btn_delete.config(state=tk.DISABLED)
        self.btn_move.config(state=tk.DISABLED)
        self.btn_copy.config(state=tk.DISABLED)
        self.log("开始扫描...")
        threading.Thread(target=self.scan_thread, args=(root_dir, exts, folders), daemon=True).start()

    def scan_thread(self, root_dir, exts, folders):
        files_found = []
        dirs_found = []
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # 检查文件夹名
                for d in dirnames[:]:  # 用切片避免迭代时修改
                    if d in folders:
                        full_dir = os.path.join(dirpath, d)
                        dirs_found.append(full_dir)
                        dirnames.remove(d)  # 不再进入该目录

                # 检查文件后缀
                for f in filenames:
                    if exts:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in exts:
                            full_file = os.path.join(dirpath, f)
                            files_found.append(full_file)
        except Exception as e:
            self.log(f"扫描出错: {e}")
            self.root.after(0, self.scan_finished, [], [])
            return

        self.root.after(0, self.scan_finished, files_found, dirs_found)

    def scan_finished(self, files, dirs):
        self.files_to_delete = files
        self.dirs_to_delete = dirs
        self.log(f"扫描完成。发现 {len(files)} 个匹配文件，{len(dirs)} 个匹配文件夹。")
        if files:
            self.log("匹配的文件示例:")
            for f in files[:10]:
                self.log(f"  {f}")
            if len(files) > 10:
                self.log(f"  ... 及其他 {len(files)-10} 个文件")
        if dirs:
            self.log("匹配的文件夹:")
            for d in dirs[:10]:
                self.log(f"  {d}")
            if len(dirs) > 10:
                self.log(f"  ... 及其他 {len(dirs)-10} 个文件夹")

        # 根据扫描结果启用按钮
        have_files = bool(files)
        have_items = bool(files or dirs)
        self.btn_delete.config(state=tk.NORMAL if have_items else tk.DISABLED)
        self.btn_move.config(state=tk.NORMAL if have_files else tk.DISABLED)
        self.btn_copy.config(state=tk.NORMAL if have_files else tk.DISABLED)
        self.btn_scan.config(state=tk.NORMAL)

    # ------------------ 删除逻辑 ------------------
    def start_delete(self):
        if not self.files_to_delete and not self.dirs_to_delete:
            messagebox.showinfo("提示", "没有可删除的项目")
            return

        msg = (f"将删除 {len(self.files_to_delete)} 个文件 和 {len(self.dirs_to_delete)} 个文件夹，"
               "此操作不可恢复！\n确定要继续吗？")
        if not messagebox.askyesno("危险操作确认", msg):
            return

        self.btn_delete.config(state=tk.DISABLED)
        self.btn_move.config(state=tk.DISABLED)
        self.btn_copy.config(state=tk.DISABLED)
        self.btn_scan.config(state=tk.DISABLED)
        self.log("开始执行删除...")
        threading.Thread(target=self.delete_thread, daemon=True).start()

    def delete_thread(self):
        success = 0
        fail = 0
        for f in self.files_to_delete:
            try:
                os.remove(f)
                self.log(f"已删除文件: {f}")
                success += 1
            except Exception as e:
                self.log(f"删除文件失败: {f} - {e}")
                fail += 1

        for d in self.dirs_to_delete:
            try:
                shutil.rmtree(d)
                self.log(f"已删除文件夹: {d}")
                success += 1
            except Exception as e:
                self.log(f"删除文件夹失败: {d} - {e}")
                fail += 1

        self.root.after(0, self.delete_finished, success, fail)

    def delete_finished(self, success, fail):
        self.log(f"删除完成。成功: {success}，失败: {fail}")
        self.btn_scan.config(state=tk.NORMAL)
        self.btn_delete.config(state=tk.DISABLED)
        self.files_to_delete = []
        self.dirs_to_delete = []

    # ------------------ 移动/复制逻辑（新增） ------------------
    def start_move_copy(self, operation):
        """operation: 'move' 或 'copy'"""
        if not self.files_to_delete:
            messagebox.showinfo("提示", "没有可移动/复制的文件")
            return

        # 选择目标目录
        dest_dir = filedialog.askdirectory(title=f"选择{'移动' if operation=='move' else '复制'}的目标文件夹")
        if not dest_dir:
            return  # 用户取消

        # 确认
        action = "移动" if operation == "move" else "复制"
        msg = f"将{action} {len(self.files_to_delete)} 个文件到：\n{dest_dir}\n确定吗？"
        if not messagebox.askyesno("确认操作", msg):
            return

        self.btn_delete.config(state=tk.DISABLED)
        self.btn_move.config(state=tk.DISABLED)
        self.btn_copy.config(state=tk.DISABLED)
        self.btn_scan.config(state=tk.DISABLED)
        self.log(f"开始{action}文件...")
        threading.Thread(target=self.move_copy_thread, args=(operation, dest_dir), daemon=True).start()

    def move_copy_thread(self, operation, dest_dir):
        success = 0
        fail = 0
        func = shutil.move if operation == "move" else shutil.copy2

        for src in self.files_to_delete:
            try:
                # 目标文件路径（保持原文件名）
                dest = os.path.join(dest_dir, os.path.basename(src))
                func(src, dest)
                self.log(f"已{operation}：{src} -> {dest}")
                success += 1
            except Exception as e:
                self.log(f"{operation}失败：{src} - {e}")
                fail += 1

        self.root.after(0, self.move_copy_finished, success, fail, operation)

    def move_copy_finished(self, success, fail, operation):
        action = "移动" if operation == "move" else "复制"
        self.log(f"{action}完成。成功: {success}，失败: {fail}")
        self.btn_scan.config(state=tk.NORMAL)
        # 操作完成后不自动清空文件列表，方便用户再次操作（例如复制到另一个地方）
        # 但按钮状态需要根据当前列表重新判断
        have_files = bool(self.files_to_delete)
        have_items = bool(self.files_to_delete or self.dirs_to_delete)
        self.btn_delete.config(state=tk.NORMAL if have_items else tk.DISABLED)
        self.btn_move.config(state=tk.NORMAL if have_files else tk.DISABLED)
        self.btn_copy.config(state=tk.NORMAL if have_files else tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = CleanerApp(root)
    root.mainloop()