from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from config import DATABASE_PATH


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
                name TEXT NOT NULL,
                department TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                created_at TEXT NOT NULL,
                FOREIGN KEY (student_id)
                    REFERENCES students(id)
                    ON DELETE CASCADE,
                UNIQUE(student_id, attendance_date)
            );

            CREATE INDEX IF NOT EXISTS idx_attendance_date
            ON attendance(attendance_date);

            CREATE INDEX IF NOT EXISTS idx_students_code
            ON students(student_code);
            """
        )


def add_student(student_code: str, name: str, department: str = "") -> int:
    student_code = student_code.strip()
    name = name.strip()
    department = department.strip()

    if not student_code:
        raise ValueError("Student ID cannot be empty.")
    if not name:
        raise ValueError("Student name cannot be empty.")

    now = datetime.now().isoformat(timespec="seconds")

    try:
        with database_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO students(student_code, name, department, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (student_code, name, department, now),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"A student with ID '{student_code}' already exists."
        ) from error


def delete_student(student_id: int) -> None:
    with database_connection() as connection:
        connection.execute(
            "DELETE FROM students WHERE id = ?",
            (student_id,),
        )


def get_student(student_id: int) -> sqlite3.Row | None:
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT id, student_code, name, department, created_at
            FROM students
            WHERE id = ?
            """,
            (student_id,),
        ).fetchone()


def get_student_by_code(student_code: str) -> sqlite3.Row | None:
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT id, student_code, name, department, created_at
            FROM students
            WHERE student_code = ? COLLATE NOCASE
            """,
            (student_code.strip(),),
        ).fetchone()


def list_students() -> list[sqlite3.Row]:
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT id, student_code, name, department, created_at
            FROM students
            ORDER BY name COLLATE NOCASE, student_code COLLATE NOCASE
            """
        ).fetchall()


def mark_attendance(
    student_id: int,
    status: str = "Present",
    when: datetime | None = None,
) -> bool:
    """
    Returns True when a new attendance record is inserted.
    Returns False when attendance already exists for that student today.
    """
    moment = when or datetime.now()
    attendance_date = moment.date().isoformat()
    attendance_time = moment.strftime("%H:%M:%S")
    created_at = moment.isoformat(timespec="seconds")

    with database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO attendance(
                student_id,
                attendance_date,
                attendance_time,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                attendance_date,
                attendance_time,
                status,
                created_at,
            ),
        )
        return cursor.rowcount == 1


def list_attendance(
    attendance_date: str | None = None,
) -> list[sqlite3.Row]:
    with database_connection() as connection:
        if attendance_date:
            return connection.execute(
                """
                SELECT
                    attendance.id,
                    students.student_code,
                    students.name,
                    students.department,
                    attendance.attendance_date,
                    attendance.attendance_time,
                    attendance.status
                FROM attendance
                INNER JOIN students
                    ON students.id = attendance.student_id
                WHERE attendance.attendance_date = ?
                ORDER BY attendance.attendance_time DESC
                """,
                (attendance_date,),
            ).fetchall()

        return connection.execute(
            """
            SELECT
                attendance.id,
                students.student_code,
                students.name,
                students.department,
                attendance.attendance_date,
                attendance.attendance_time,
                attendance.status
            FROM attendance
            INNER JOIN students
                ON students.id = attendance.student_id
            ORDER BY
                attendance.attendance_date DESC,
                attendance.attendance_time DESC
            """
        ).fetchall()


def today_attendance() -> list[sqlite3.Row]:
    return list_attendance(date.today().isoformat())


def export_attendance_csv(
    output_path: Path,
    attendance_date: str | None = None,
) -> int:
    records = list_attendance(attendance_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Student ID",
                "Name",
                "Department",
                "Date",
                "Time",
                "Status",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["student_code"],
                    record["name"],
                    record["department"],
                    record["attendance_date"],
                    record["attendance_time"],
                    record["status"],
                ]
            )

    return len(records)
