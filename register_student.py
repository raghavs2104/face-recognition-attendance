from __future__ import annotations

import argparse

from database import initialize_database
from vision import capture_student_faces


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a student and capture face samples."
    )
    parser.add_argument("--id", required=True, help="Student ID or roll number")
    parser.add_argument("--name", required=True, help="Student full name")
    parser.add_argument(
        "--department",
        default="",
        help="Department or class",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default: 0",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=40,
        help="Number of face samples. Default: 40",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    initialize_database()

    student_id, saved_count = capture_student_faces(
        student_code=args.id,
        name=args.name,
        department=args.department,
        camera_index=args.camera,
        sample_count=args.samples,
    )

    print(
        f"Registration complete. Database ID={student_id}, "
        f"samples={saved_count}"
    )
    print("Run train_model.py before taking attendance.")


if __name__ == "__main__":
    main()
