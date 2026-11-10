from __future__ import annotations

import json
import platform
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from config import (
    ATTENDANCE_STATUS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    FACE_HEIGHT,
    FACE_MIN_NEIGHBORS,
    FACE_MIN_SIZE,
    FACE_SCALE_FACTOR,
    FACE_SIZE,
    FACE_WIDTH,
    FACES_DIR,
    LBPH_DISTANCE_THRESHOLD,
    MODEL_INFO_PATH,
    MODEL_PATH,
    MODELS_DIR,
    REGISTRATION_SAMPLE_COUNT,
    REQUIRED_CONSECUTIVE_MATCHES,
)
from database import (
    add_student,
    delete_student,
    get_student,
    list_students,
    mark_attendance,
)


def ensure_opencv_face_module() -> None:
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV's face module is missing.\n\n"
            "Uninstall conflicting OpenCV packages and install:\n"
            "pip uninstall opencv-python opencv-python-headless "
            "opencv-contrib-python opencv-contrib-python-headless -y\n"
            "pip install opencv-contrib-python"
        )

    if not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPHFaceRecognizer is unavailable in this OpenCV build. "
            "Install opencv-contrib-python."
        )


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def open_camera(camera_index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Prefer DirectShow on Windows, then fall back to OpenCV's default backend.
    """
    if platform.system() == "Windows":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            return camera
        camera.release()

    camera = cv2.VideoCapture(camera_index)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return camera

    camera.release()
    raise RuntimeError(
        f"Could not open camera index {camera_index}. "
        "Close other camera apps or try camera index 1."
    )


def detect_faces(
    detector: cv2.CascadeClassifier,
    grayscale_image: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    faces = detector.detectMultiScale(
        grayscale_image,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return sorted(
        (tuple(map(int, face)) for face in faces),
        key=lambda rectangle: rectangle[2] * rectangle[3],
        reverse=True,
    )


def normalize_face(face_gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return equalized


def _draw_text_box(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (10, 10),
) -> None:
    x, y = origin
    line_height = 28
    box_height = 15 + line_height * len(lines)
    box_width = min(frame.shape[1] - x - 10, 760)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + box_width, y + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line[:85],
            (x + 10, y + 25 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def capture_student_faces(
    student_code: str,
    name: str,
    department: str = "",
    camera_index: int = CAMERA_INDEX,
    sample_count: int = REGISTRATION_SAMPLE_COUNT,
) -> tuple[int, int]:
    """
    Creates the student record and collects face images.

    Returns:
        (database_student_id, number_of_saved_samples)
    """
    if sample_count < 10:
        raise ValueError("Collect at least 10 samples.")

    detector = create_face_detector()
    student_id = add_student(student_code, name, department)
    student_folder = FACES_DIR / str(student_id)

    if student_folder.exists():
        shutil.rmtree(student_folder)
    student_folder.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    cancelled = False
    camera: cv2.VideoCapture | None = None

    try:
        camera = open_camera(camera_index)
        last_capture_time = 0.0
        minimum_capture_interval = 0.09

        while saved_count < sample_count:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(detector, grayscale)

            message = "No face detected"
            rectangle_color = (0, 180, 255)

            if faces:
                x, y, width, height = faces[0]
                face = grayscale[y : y + height, x : x + width]
                normalized = normalize_face(face)

                rectangle_color = (0, 200, 0)
                message = "Face detected - move slightly between samples"

                current_time = time.monotonic()
                if (
                    current_time - last_capture_time
                    >= minimum_capture_interval
                ):
                    saved_count += 1
                    output_path = (
                        student_folder / f"sample_{saved_count:03d}.jpg"
                    )
                    if not cv2.imwrite(str(output_path), normalized):
                        raise RuntimeError(
                            f"Could not save image: {output_path}"
                        )
                    last_capture_time = current_time

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    rectangle_color,
                    3,
                )

            _draw_text_box(
                frame,
                [
                    f"Registering: {name} ({student_code})",
                    f"Saved samples: {saved_count}/{sample_count}",
                    message,
                    "Look at the camera. Turn slightly left/right. Press Q to cancel.",
                ],
            )

            cv2.imshow("Register Face", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cancelled = True
                break

        if cancelled:
            raise RuntimeError("Face registration was cancelled.")

        return student_id, saved_count

    except Exception:
        shutil.rmtree(student_folder, ignore_errors=True)
        delete_student(student_id)
        raise
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


def _iter_training_images() -> tuple[list[np.ndarray], list[int], dict[int, int]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    counts: dict[int, int] = defaultdict(int)

    students = list_students()
    if not students:
        raise RuntimeError("No registered students were found.")

    for student in students:
        student_id = int(student["id"])
        student_folder = FACES_DIR / str(student_id)
        if not student_folder.exists():
            continue

        for image_path in sorted(student_folder.glob("*.jpg")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = normalize_face(image)
            images.append(image)
            labels.append(student_id)
            counts[student_id] += 1

    if not images:
        raise RuntimeError(
            "No face samples were found. Register at least one student first."
        )

    return images, labels, dict(counts)


def train_lbph_model() -> dict[str, object]:
    ensure_opencv_face_module()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    images, labels, counts = _iter_training_images()

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=LBPH_DISTANCE_THRESHOLD,
    )
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))

    model_info: dict[str, object] = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "student_count": len(counts),
        "image_count": len(images),
        "sample_counts": {str(key): value for key, value in counts.items()},
        "face_size": [FACE_WIDTH, FACE_HEIGHT],
        "distance_threshold": LBPH_DISTANCE_THRESHOLD,
    }

    MODEL_INFO_PATH.write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )
    return model_info


def load_recognizer():
    ensure_opencv_face_module()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained face model does not exist.\n"
            "Register students and click 'Train Model' first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    return recognizer


def distance_to_match_score(distance: float) -> float:
    """
    A display-only heuristic, not a calibrated probability.
    Lower LBPH distance means a better match.
    """
    return max(0.0, min(100.0, 100.0 - distance))


def run_attendance_camera(
    camera_index: int = CAMERA_INDEX,
    status_callback: Callable[[str], None] | None = None,
) -> set[int]:
    detector = create_face_detector()
    recognizer = load_recognizer()
    students = {
        int(student["id"]): student
        for student in list_students()
    }

    camera: cv2.VideoCapture | None = None
    consecutive_matches: dict[int, int] = defaultdict(int)
    marked_this_session: set[int] = set()
    last_seen_student_ids: set[int] = set()

    def report(message: str) -> None:
        print(message)
        if status_callback:
            status_callback(message)

    try:
        camera = open_camera(camera_index)
        report("Attendance camera started. Press Q to stop.")

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = detect_faces(detector, grayscale)
            seen_in_current_frame: set[int] = set()

            for x, y, width, height in detected_faces:
                face = grayscale[y : y + height, x : x + width]
                normalized = normalize_face(face)

                predicted_id, distance = recognizer.predict(normalized)
                predicted_id = int(predicted_id)
                distance = float(distance)

                student = students.get(predicted_id)
                is_known = (
                    student is not None
                    and distance <= LBPH_DISTANCE_THRESHOLD
                )

                if is_known:
                    seen_in_current_frame.add(predicted_id)
                    consecutive_matches[predicted_id] += 1

                    name = str(student["name"])
                    student_code = str(student["student_code"])
                    score = distance_to_match_score(distance)

                    label = f"{name} | {student_code}"
                    second_line = (
                        f"Match score: {score:.1f} "
                        f"({consecutive_matches[predicted_id]}/"
                        f"{REQUIRED_CONSECUTIVE_MATCHES})"
                    )
                    rectangle_color = (0, 200, 0)

                    if (
                        consecutive_matches[predicted_id]
                        >= REQUIRED_CONSECUTIVE_MATCHES
                        and predicted_id not in marked_this_session
                    ):
                        inserted = mark_attendance(
                            predicted_id,
                            ATTENDANCE_STATUS,
                        )
                        marked_this_session.add(predicted_id)

                        if inserted:
                            report(
                                f"Attendance marked: {name} ({student_code})"
                            )
                        else:
                            report(
                                f"Already marked today: {name} "
                                f"({student_code})"
                            )
                else:
                    label = "Unknown"
                    second_line = f"LBPH distance: {distance:.1f}"
                    rectangle_color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    rectangle_color,
                    3,
                )

                text_y = max(25, y - 35)
                cv2.putText(
                    frame,
                    label[:55],
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    rectangle_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    second_line[:55],
                    (x, text_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    rectangle_color,
                    2,
                    cv2.LINE_AA,
                )

            # Reset streaks when a recognized person disappears from the frame.
            disappeared = last_seen_student_ids - seen_in_current_frame
            for student_id in disappeared:
                consecutive_matches[student_id] = 0
            last_seen_student_ids = seen_in_current_frame

            _draw_text_box(
                frame,
                [
                    "Face Recognition Attendance",
                    f"Marked this session: {len(marked_this_session)}",
                    "Press Q to close",
                ],
            )

            cv2.imshow("Face Recognition Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        return marked_this_session
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
