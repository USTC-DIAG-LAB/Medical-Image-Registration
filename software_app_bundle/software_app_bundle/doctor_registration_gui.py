import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


MODALITIES = ("CT", "MRI", "PET", "SPECT", "CBCT", "US", "XRAY", "OCT", "MG")


class DoctorRegistrationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("医学图像配准工具")
        self.root.geometry("880x700")

        self.project_root = Path(__file__).resolve().parent
        self.backend_script = self.project_root / "one_click_registration.py"
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker = None

        self.moving_image_var = tk.StringVar()
        self.fixed_image_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.case_name_var = tk.StringVar()
        self.moving_modality_var = tk.StringVar(value="CT")
        self.fixed_modality_var = tk.StringVar(value="MRI")
        self.overwrite_var = tk.BooleanVar(value=True)
        self.matlab_bin_var = tk.StringVar(value="matlab")

        self._build_ui()
        self.root.after(150, self._poll_logs)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="医学图像配准工具（医生版）", font=("Arial", 18, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            container,
            text="选择 moving 图像、fixed 图像和输出文件夹，然后点击“开始配准”。",
            font=("Arial", 11),
        )
        subtitle.pack(anchor="w", pady=(6, 16))

        form = ttk.Frame(container)
        form.pack(fill=tk.X)

        self._add_path_row(form, 0, "Moving 图像", self.moving_image_var, self._choose_moving_image)
        self._add_path_row(form, 1, "Fixed 图像", self.fixed_image_var, self._choose_fixed_image)
        self._add_path_row(form, 2, "输出文件夹", self.output_dir_var, self._choose_output_dir)

        ttk.Label(form, text="Moving 模态").grid(row=3, column=0, sticky="w", padx=(0, 12), pady=10)
        ttk.Combobox(form, textvariable=self.moving_modality_var, values=MODALITIES, state="readonly", width=14).grid(
            row=3, column=1, sticky="w", pady=10
        )

        ttk.Label(form, text="Fixed 模态").grid(row=3, column=2, sticky="w", padx=(20, 12), pady=10)
        ttk.Combobox(form, textvariable=self.fixed_modality_var, values=MODALITIES, state="readonly", width=14).grid(
            row=3, column=3, sticky="w", pady=10
        )

        ttk.Label(form, text="病例名称（可选）").grid(row=4, column=0, sticky="w", padx=(0, 12), pady=10)
        ttk.Entry(form, textvariable=self.case_name_var, width=28).grid(row=4, column=1, sticky="we", pady=10)

        ttk.Label(form, text="MATLAB 命令").grid(row=4, column=2, sticky="w", padx=(20, 12), pady=10)
        ttk.Entry(form, textvariable=self.matlab_bin_var, width=22).grid(row=4, column=3, sticky="w", pady=10)

        ttk.Checkbutton(form, text="覆盖已有输出", variable=self.overwrite_var).grid(
            row=5, column=0, sticky="w", pady=(6, 2)
        )

        for col in range(4):
            form.columnconfigure(col, weight=1 if col in {1, 3} else 0)

        button_bar = ttk.Frame(container)
        button_bar.pack(fill=tk.X, pady=(18, 12))

        self.run_button = ttk.Button(button_bar, text="开始配准", command=self._start_registration)
        self.run_button.pack(side=tk.LEFT)

        ttk.Button(button_bar, text="清空日志", command=self._clear_log).pack(side=tk.LEFT, padx=(10, 0))

        hint = ttk.Label(
            container,
            text="输出会自动包含：配准后的 3D 图像，以及 self 版本切片图。",
            font=("Arial", 10),
        )
        hint.pack(anchor="w", pady=(0, 8))

        self.log_box = scrolledtext.ScrolledText(container, wrap=tk.WORD, font=("Courier", 10), height=22)
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.configure(state=tk.DISABLED)

    def _add_path_row(self, parent, row: int, label: str, variable: tk.StringVar, callback) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=10)
        ttk.Entry(parent, textvariable=variable, width=70).grid(row=row, column=1, columnspan=2, sticky="we", pady=10)
        ttk.Button(parent, text="浏览...", command=callback).grid(row=row, column=3, sticky="e", pady=10)

    def _choose_moving_image(self) -> None:
        path = filedialog.askopenfilename(title="选择 Moving 图像")
        if path:
            self.moving_image_var.set(path)

    def _choose_fixed_image(self) -> None:
        path = filedialog.askopenfilename(title="选择 Fixed 图像")
        if path:
            self.fixed_image_var.set(path)

    def _choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_dir_var.set(path)

    def _clear_log(self) -> None:
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _validate_inputs(self) -> bool:
        moving = Path(self.moving_image_var.get().strip())
        fixed = Path(self.fixed_image_var.get().strip())
        output_dir = self.output_dir_var.get().strip()

        if not moving.exists():
            messagebox.showerror("输入错误", "请选择有效的 Moving 图像。")
            return False
        if not fixed.exists():
            messagebox.showerror("输入错误", "请选择有效的 Fixed 图像。")
            return False
        if not output_dir:
            messagebox.showerror("输入错误", "请选择输出文件夹。")
            return False
        return True

    def _build_command(self):
        cmd = [
            sys.executable,
            str(self.backend_script),
            "--moving-image",
            self.moving_image_var.get().strip(),
            "--fixed-image",
            self.fixed_image_var.get().strip(),
            "--moving-modality",
            self.moving_modality_var.get().strip(),
            "--fixed-modality",
            self.fixed_modality_var.get().strip(),
            "--output-dir",
            self.output_dir_var.get().strip(),
            "--matlab-bin",
            self.matlab_bin_var.get().strip() or "matlab",
        ]
        case_name = self.case_name_var.get().strip()
        if case_name:
            cmd.extend(["--case-name", case_name])
        if self.overwrite_var.get():
            cmd.append("--overwrite")
        return cmd

    def _start_registration(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在运行", "当前已有任务正在运行，请等待完成。")
            return
        if not self._validate_inputs():
            return

        self._append_log("\n============================================================\n")
        self._append_log("开始新的配准任务\n")
        self._append_log("============================================================\n")

        cmd = self._build_command()
        self.run_button.configure(state=tk.DISABLED)
        self.worker = threading.Thread(target=self._run_backend, args=(cmd,), daemon=True)
        self.worker.start()

    def _run_backend(self, cmd) -> None:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(line)
            return_code = process.wait()
            if return_code == 0:
                self.log_queue.put("\n任务完成。\n")
            else:
                self.log_queue.put(f"\n任务失败，返回码：{return_code}\n")
        except Exception as exc:
            self.log_queue.put(f"\n启动失败：{exc}\n")
        finally:
            self.log_queue.put("__TASK_DONE__")

    def _poll_logs(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__TASK_DONE__":
                self.run_button.configure(state=tk.NORMAL)
            else:
                self._append_log(item)
        self.root.after(150, self._poll_logs)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    DoctorRegistrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
