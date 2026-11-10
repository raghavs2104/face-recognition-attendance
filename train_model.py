from __future__ import annotations

from database import initialize_database
from vision import train_lbph_model


def main() -> None:
    initialize_database()
    model_info = train_lbph_model()

    print("LBPH training complete.")
    print(f"Students:    {model_info['student_count']}")
    print(f"Face images:{model_info['image_count']}")
    print(f"Trained at:  {model_info['trained_at']}")


if __name__ == "__main__":
    main()
