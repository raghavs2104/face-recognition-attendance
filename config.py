from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
DATABASE_PATH = DATA_DIR / "attendance.db"

MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "lbph_face_model.yml"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

EXPORTS_DIR = PROJECT_DIR / "exports"

FACE_WIDTH = 200
FACE_HEIGHT = 200
FACE_SIZE = (FACE_WIDTH, FACE_HEIGHT)

CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

# Number of usable face samples collected during registration.
REGISTRATION_SAMPLE_COUNT = 40

# Haar detector settings.
FACE_SCALE_FACTOR = 1.2
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (90, 90)

# LBPH returns a distance: lower is a better match.
# Values above this threshold are treated as unknown.
LBPH_DISTANCE_THRESHOLD = 68.0

# Require the same recognized person across several frames before attendance
# is written. This reduces accidental one-frame matches.
REQUIRED_CONSECUTIVE_MATCHES = 5

# A record with the same student and date is not inserted twice.
ATTENDANCE_STATUS = "Present"
