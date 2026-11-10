from __future__ import annotations

import shutil

from config import (
    DATABASE_PATH,
    EXPORTS_DIR,
    FACES_DIR,
    MODEL_INFO_PATH,
    MODEL_PATH,
)
from database import initialize_database


def main() -> None:
    confirmation = input(
        "This deletes all students, face images, models, attendance and "
        "exports. Type DELETE to continue: "
    )
    if confirmation != "DELETE":
        print("Reset cancelled.")
        return

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    shutil.rmtree(FACES_DIR, ignore_errors=True)
    shutil.rmtree(EXPORTS_DIR, ignore_errors=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in (MODEL_PATH, MODEL_INFO_PATH):
        if path.exists():
            path.unlink()

    initialize_database()
    print("Project data reset successfully.")


if __name__ == "__main__":
    main()
