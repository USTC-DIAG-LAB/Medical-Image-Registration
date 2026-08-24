import importlib.util
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ModuleNotFoundError as exc:
    raise SystemExit(
        "FOTMIR GUI requires Tkinter, but this Python installation does not provide it. "
        f"Use a Python environment with Tk support. Current interpreter: {sys.executable}"
    ) from exc


MODALITIES = ("CT", "MRI", "PET", "SPECT", "CBCT", "US", "XRAY", "OCT", "MG")
SUPPORTED_SUFFIXES = (
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".nrrd",
    ".img",
    ".hdr",
    ".mnc",
    ".mnc2",
    ".dcm",
    ".ima",
)
REQUIRED_PYTHON_MODULES = {
    "numpy": "NumPy",
    "scipy": "SciPy",
    "SimpleITK": "SimpleITK",
    "matplotlib": "Matplotlib",
}


def is_supported_image_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def find_executable(command: str) -> Optional[str]:
    value = command.strip()
    if not value:
        return None

    expanded = Path(value).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        return str(expanded.resolve()) if expanded.is_file() and os.access(expanded, os.X_OK) else None
    return shutil.which(value)


class DoctorRegistrationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("医学图像配准工具")
        self.root.geometry("880x700")

        self.project_root = Path(__file__).resolve().parent
        self.backend_script = self.project_root / "one_click_registration.py"
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker = None
        self.process: Optional[subprocess.Popen] = None
        self.cancel_requested = False
        self.closing = False

        self.moving_image_var = tk.StringVar()
        self.fixed_image_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.case_name_var = tk.StringVar()
        self.moving_modality_var = tk.StringVar(value="CT")
        self.fixed_modality_var = tk.StringVar(value="MRI")
        self.overwrite_var = tk.BooleanVar(value=False)
        self.matlab_bin_var = tk.StringVar(value="matlab")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._poll_logs)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="FOTMIR 医学图像配准工具", font=("Arial", 18, "bold"))
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

        ttk.Label(form, text="MATLAB 可执行文件").grid(row=4, column=2, sticky="w", padx=(20, 12), pady=10)
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

        self.cancel_button = ttk.Button(button_bar, text="取消任务", command=self._cancel_registration, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Button(button_bar, text="检查环境", command=self._check_environment_interactive).pack(side=tk.LEFT, padx=(10, 0))

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
        moving_text = self.moving_image_var.get().strip()
        fixed_text = self.fixed_image_var.get().strip()
        output_text = self.output_dir_var.get().strip()

        if not moving_text:
            messagebox.showerror("输入错误", "请选择 Moving 图像。")
            return False
        if not fixed_text:
            messagebox.showerror("输入错误", "请选择 Fixed 图像。")
            return False
        if not output_text:
            messagebox.showerror("输入错误", "请选择输出文件夹。")
            return False

        moving = Path(moving_text).expanduser()
        fixed = Path(fixed_text).expanduser()
        output_dir = Path(output_text).expanduser()

        if not moving.is_file():
            messagebox.showerror("输入错误", "请选择有效的 Moving 图像。")
            return False
        if not fixed.is_file():
            messagebox.showerror("输入错误", "请选择有效的 Fixed 图像。")
            return False
        if not is_supported_image_file(moving):
            messagebox.showerror("输入错误", f"Moving 图像格式不受支持：{moving.name}")
            return False
        if not is_supported_image_file(fixed):
            messagebox.showerror("输入错误", f"Fixed 图像格式不受支持：{fixed.name}")
            return False
        if moving.resolve() == fixed.resolve():
            messagebox.showerror("输入错误", "Moving 图像和 Fixed 图像不能是同一文件。")
            return False
        if output_dir.exists() and not output_dir.is_dir():
            messagebox.showerror("输入错误", "输出路径已存在，但不是文件夹。")
            return False
        return True

    def _environment_issues(self) -> List[str]:
        issues: List[str] = []
        if not self.backend_script.is_file():
            issues.append(f"未找到后端脚本：{self.backend_script}")

        for module_name, display_name in REQUIRED_PYTHON_MODULES.items():
            if importlib.util.find_spec(module_name) is None:
                issues.append(f"当前 Python 缺少 {display_name}：{sys.executable}")

        matlab_value = self.matlab_bin_var.get().strip() or "matlab"
        if find_executable(matlab_value) is None:
            issues.append(
                "未找到 MATLAB 可执行文件。请安装 MATLAB，或填写类似 "
                "'/Applications/MATLAB_R2026a.app/bin/matlab' 的完整路径。"
            )
        return issues

    def _check_environment(self, show_success: bool) -> bool:
        issues = self._environment_issues()
        self._append_log(f"\nPython: {sys.executable}\n")
        if issues:
            details = "\n".join(f"- {issue}" for issue in issues)
            self._append_log("环境检查未通过：\n" + details + "\n")
            messagebox.showerror("环境检查未通过", details)
            return False

        self._append_log("环境检查通过。\n")
        if show_success:
            messagebox.showinfo("环境检查", "Python 依赖、后端脚本和 MATLAB 可执行文件均已找到。")
        return True

    def _check_environment_interactive(self) -> None:
        self._check_environment(show_success=True)

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
        if not self._check_environment(show_success=False):
            return

        output_dir = Path(self.output_dir_var.get().strip()).expanduser()
        if self.overwrite_var.get() and output_dir.exists():
            confirmed = messagebox.askyesno(
                "确认覆盖",
                "将只替换该输出目录中的同名病例结果，不会删除整个输出目录。是否继续？",
            )
            if not confirmed:
                return

        self._append_log("\n============================================================\n")
        self._append_log("开始新的配准任务\n")
        self._append_log("============================================================\n")

        cmd = self._build_command()
        self.cancel_requested = False
        self.run_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self.worker = threading.Thread(target=self._run_backend, args=(cmd,), daemon=True)
        self.worker.start()

    def _run_backend(self, cmd) -> None:
        try:
            popen_options = {}
            if os.name == "nt":
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_options["start_new_session"] = True

            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_options,
            )
            self.process = process
            if self.cancel_requested:
                self._terminate_process(process)
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(line)
            return_code = process.wait()
            if self.cancel_requested:
                self.log_queue.put("\n任务已取消。\n")
            elif return_code == 0:
                self.log_queue.put("\n任务完成。\n")
            else:
                self.log_queue.put(f"\n任务失败，返回码：{return_code}\n")
        except Exception as exc:
            self.log_queue.put(f"\n启动失败：{exc}\n")
        finally:
            self.process = None
            self.log_queue.put("__TASK_DONE__")

    def _terminate_process(self, process: subprocess.Popen, force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                if force:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process_group = os.getpgid(process.pid)
                os.killpg(process_group, signal.SIGKILL if force else signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                process.kill() if force else process.terminate()
            except OSError:
                pass

    def _cancel_registration(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        if not messagebox.askyesno("取消任务", "确定要终止当前配准任务吗？"):
            return

        self.cancel_requested = True
        self.cancel_button.configure(state=tk.DISABLED)
        self._append_log("\n正在终止当前任务...\n")
        if self.process is not None:
            self._terminate_process(self.process)
        self.root.after(3000, self._force_kill_if_needed)

    def _force_kill_if_needed(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._terminate_process(self.process, force=True)

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("退出程序", "配准任务仍在运行。终止任务并退出吗？"):
                return
            self.closing = True
            self.cancel_requested = True
            if self.process is not None:
                self._terminate_process(self.process)
            self.root.after(500, self._finish_close)
            return
        self.root.destroy()

    def _finish_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._terminate_process(self.process, force=True)
        self.root.destroy()

    def _poll_logs(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__TASK_DONE__":
                self.run_button.configure(state=tk.NORMAL)
                self.cancel_button.configure(state=tk.DISABLED)
            else:
                self._append_log(item)
        if not self.closing:
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
