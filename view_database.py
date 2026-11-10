from __future__ import annotations

from database import initialize_database, list_attendance, list_students


def main() -> None:
    initialize_database()

    students = list_students()
    print("\nREGISTERED STUDENTS")
    print("=" * 80)
    if not students:
        print("No students registered.")
    else:
        for student in students:
            print(
                f"{student['student_code']} | {student['name']} | "
                f"{student['department']} | {student['created_at']}"
            )

    records = list_attendance()
    print("\nATTENDANCE")
    print("=" * 80)
    if not records:
        print("No attendance records.")
    else:
        for record in records:
            print(
                f"{record['attendance_date']} {record['attendance_time']} | "
                f"{record['student_code']} | {record['name']} | "
                f"{record['status']}"
            )


if __name__ == "__main__":
    main()
