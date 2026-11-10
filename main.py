from __future__ import annotations

import threading
from datetime import date, datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from config import CAMERA_INDEX, EXPORTS_DIR, MODEL_INFO_PATH
from database import (
    export_attendance_csv,
    initialize_database,
    list_attendance,
    list_students,
)
from vision import (
    capture_student_faces,
    run_attendance_camera,
    train_lbph_model,
)


class AttendanceApplication:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1100x720")
        self.root.minsize(920, 640)

        self.is_busy = False
        self.camera_index_var = tk.IntVar(value=CAMERA_INDEX)
        self.status_var = tk.StringVar(value="Ready")

        initialize_database()
        self._configure_style()
        self._build_interface()
        self.refresh_students()
        self.refresh_attendance()

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
        )

    def _build_interface(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            outer,
            text="Face Recognition Attendance System",
            style="Title.TLabel",
        )
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            outer,
            text=(
                "OpenCV Haar face detection + LBPH recognition + SQLite. "
                "Use only with informed consent."
            ),
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor=tk.W, pady=(2, 14))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill=tk.X, pady=(0, 12))

        ttk.Button(
            toolbar,
            text="Register Student",
            command=self.open_registration_dialog,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            toolbar,
            text="Train Model",
            command=self.start_model_training,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(
            toolbar,
            text="Start Attendance",
            command=self.start_attendance,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(
            toolbar,
            text="Export CSV",
            command=self.export_csv,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(
            toolbar,
            text="Refresh",
            command=self.refresh_all,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Label(toolbar, text="Camera:").pack(side=tk.LEFT, padx=(22, 4))
        camera_spinbox = ttk.Spinbox(
            toolbar,
            from_=0,
            to=9,
            width=4,
            textvariable=self.camera_index_var,
        )
        camera_spinbox.pack(side=tk.LEFT)

        notebook = ttk.Notebook(outer)
        notebook.pack(fill=tk.BOTH, expand=True)

        students_tab = ttk.Frame(notebook, padding=10)
        attendance_tab = ttk.Frame(notebook, padding=10)
        notebook.add(students_tab, text="Registered Students")
        notebook.add(attendance_tab, text="Attendance Records")

        self.students_tree = self._create_students_table(students_tab)
        self.attendance_tree = self._create_attendance_table(attendance_tab)

        status_bar = ttk.Label(
            outer,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(8, 5),
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _create_students_table(self, parent: ttk.Frame) -> ttk.Treeview:
        columns = ("student_code", "name", "department", "created_at")
        tree = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        headings = {
            "student_code": "Student ID",
            "name": "Name",
            "department": "Department",
            "created_at": "Registered At",
        }
        widths = {
            "student_code": 150,
            "name": 260,
            "department": 220,
            "created_at": 190,
        }

        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.W)

        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=tree.yview,
        )
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def _create_attendance_table(self, parent: ttk.Frame) -> ttk.Treeview:
        columns = (
            "student_code",
            "name",
            "department",
            "date",
            "time",
            "status",
        )
        tree = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        headings = {
            "student_code": "Student ID",
            "name": "Name",
            "department": "Department",
            "date": "Date",
            "time": "Time",
            "status": "Status",
        }
        widths = {
            "student_code": 135,
            "name": 235,
            "department": 200,
            "date": 110,
            "time": 100,
            "status": 100,
        }

        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.W)

        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=tree.yview,
        )
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def set_status(self, message: str) -> None:
        self.root.after(0, self.status_var.set, message)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self.is_busy = busy
        if message:
            self.set_status(message)

    def _run_background(
        self,
        work,
        success_message: str,
        after_success=None,
    ) -> None:
        if self.is_busy:
            messagebox.showwarning(
                "Busy",
                "Another operation is already running.",
            )
            return

        self._set_busy(True, "Working...")

        def worker() -> None:
            try:
                result = work()
            except Exception as error:
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Error", str(error)),
                )
                self.set_status(f"Error: {error}")
            else:
                if after_success:
                    self.root.after(0, lambda: after_success(result))
                self.set_status(success_message)
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def open_registration_dialog(self) -> None:
        if self.is_busy:
            messagebox.showwarning(
                "Busy",
                "Finish the current operation first.",
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Register Student")
        dialog.geometry("460x300")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=22)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Register a New Student",
            style="Heading.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 16))

        student_id_var = tk.StringVar()
        name_var = tk.StringVar()
        department_var = tk.StringVar()

        fields = [
            ("Student ID:", student_id_var),
            ("Full Name:", name_var),
            ("Department:", department_var),
        ]

        for row_index, (label, variable) in enumerate(fields, start=1):
            ttk.Label(frame, text=label).grid(
                row=row_index,
                column=0,
                sticky=tk.W,
                pady=8,
                padx=(0, 12),
            )
            entry = ttk.Entry(frame, textvariable=variable, width=34)
            entry.grid(row=row_index, column=1, sticky=tk.EW, pady=8)
            if row_index == 1:
                entry.focus_set()

        instruction = ttk.Label(
            frame,
            text=(
                "After clicking Start Capture, look at the camera and move "
                "your head slightly left and right. Press Q to cancel."
            ),
            wraplength=410,
            justify=tk.LEFT,
        )
        instruction.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky=tk.W,
            pady=(12, 16),
        )

        def begin_capture() -> None:
            student_code = student_id_var.get().strip()
            name = name_var.get().strip()
            department = department_var.get().strip()

            if not student_code or not name:
                messagebox.showwarning(
                    "Missing details",
                    "Student ID and full name are required.",
                    parent=dialog,
                )
                return

            camera_index = int(self.camera_index_var.get())
            dialog.destroy()

            def work():
                return capture_student_faces(
                    student_code=student_code,
                    name=name,
                    department=department,
                    camera_index=camera_index,
                )

            def after_success(result) -> None:
                student_database_id, sample_count = result
                self.refresh_students()
                messagebox.showinfo(
                    "Registration complete",
                    f"Saved {sample_count} face samples.\n\n"
                    "Click 'Train Model' before taking attendance.",
                )

            self._run_background(
                work,
                success_message=f"Registered {name}. Train the model next.",
                after_success=after_success,
            )

        button_frame = ttk.Frame(frame)
        button_frame.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky=tk.E,
        )
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            button_frame,
            text="Start Capture",
            command=begin_capture,
        ).pack(side=tk.LEFT)

        frame.columnconfigure(1, weight=1)

    def start_model_training(self) -> None:
        def after_success(info: dict[str, object]) -> None:
            messagebox.showinfo(
                "Training complete",
                "LBPH model trained successfully.\n\n"
                f"Students: {info['student_count']}\n"
                f"Face images: {info['image_count']}",
            )

        self._run_background(
            train_lbph_model,
            success_message="Model training complete.",
            after_success=after_success,
        )

    def start_attendance(self) -> None:
        camera_index = int(self.camera_index_var.get())

        def work():
            return run_attendance_camera(
                camera_index=camera_index,
                status_callback=self.set_status,
            )

        def after_success(marked_student_ids: set[int]) -> None:
            self.refresh_attendance()
            messagebox.showinfo(
                "Attendance session ended",
                f"Recognized during this session: "
                f"{len(marked_student_ids)} student(s).",
            )

        self._run_background(
            work,
            success_message="Attendance camera closed.",
            after_success=after_success,
        )

    def refresh_students(self) -> None:
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)

        for student in list_students():
            self.students_tree.insert(
                "",
                tk.END,
                values=(
                    student["student_code"],
                    student["name"],
                    student["department"],
                    student["created_at"],
                ),
            )

    def refresh_attendance(self) -> None:
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        for record in list_attendance():
            self.attendance_tree.insert(
                "",
                tk.END,
                values=(
                    record["student_code"],
                    record["name"],
                    record["department"],
                    record["attendance_date"],
                    record["attendance_time"],
                    record["status"],
                ),
            )

    def refresh_all(self) -> None:
        self.refresh_students()
        self.refresh_attendance()
        self.set_status("Tables refreshed.")

    def export_csv(self) -> None:
        default_name = (
            f"attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

        selected_path = filedialog.asksaveasfilename(
            title="Export attendance to CSV",
            initialdir=EXPORTS_DIR,
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")],
        )
        if not selected_path:
            return

        try:
            count = export_attendance_csv(Path(selected_path))
        except Exception as error:
            messagebox.showerror("Export failed", str(error))
            return

        self.set_status(f"Exported {count} record(s) to CSV.")
        messagebox.showinfo(
            "Export complete",
            f"Exported {count} attendance record(s).",
        )


def main() -> None:
    root = tk.Tk()
    AttendanceApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
