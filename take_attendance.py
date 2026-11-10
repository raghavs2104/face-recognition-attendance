from __future__ import annotations

import argparse

from database import initialize_database
from vision import run_attendance_camera


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start live face-recognition attendance."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default: 0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    initialize_database()
    marked = run_attendance_camera(camera_index=args.camera)
    print(f"Session finished. Recognized {len(marked)} student(s).")


if __name__ == "__main__":
    main()
