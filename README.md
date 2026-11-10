# Face Recognition Attendance System using OpenCV

A complete PyCharm project using:

- Python
- OpenCV Haar Cascade for face detection
- OpenCV LBPH for face recognition
- SQLite for students and attendance
- Tkinter desktop interface
- CSV export

## Features

- Register a student with ID, name and department
- Capture 40 normalized face samples using a webcam
- Train an LBPH face-recognition model
- Recognize multiple registered faces from a live camera
- Require several consecutive matches before marking attendance
- Prevent duplicate attendance for the same student on the same date
- View registered students and attendance inside the GUI
- Export attendance to CSV
- Command-line scripts are also included

## Important limitations

This is appropriate for a college project or controlled classroom demonstration.
It is not production-grade biometric security.

- LBPH recognition can fail with poor lighting, large pose changes, masks,
  blurry video or similar-looking faces.
- The system has no liveness or anti-spoofing check. A photograph might fool it.
- The displayed "match score" is only a transformed LBPH distance. It is not a
  true probability.
- Collect and process face data only with informed consent. Protect the
  `data/faces` folder and `attendance.db` because they contain personal data.

## Recommended setup

- Windows 10 or 11
- Python 3.10 or 3.11, 64-bit
- PyCharm
- Working webcam

## 1. Open the project in PyCharm

1. Extract the ZIP file.
2. Open PyCharm.
3. Select **File > Open**.
4. Select the `face_recognition_attendance_opencv` folder.
5. Create a Python 3.10 or 3.11 virtual environment.

## 2. Install dependencies

Open the PyCharm terminal in the project folder:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do **not** install `opencv-python` separately. This project needs
`opencv-contrib-python` because the LBPH recognizer is inside the contributed
`cv2.face` module.

Test the installation:

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'face'))"
```

The final value should be:

```text
True
```

### Fix: `module 'cv2' has no attribute 'face'`

Conflicting OpenCV packages are installed. Run:

```bash
pip uninstall opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless -y
pip install opencv-contrib-python
```

Restart PyCharm after reinstalling.

## 3. Run the GUI

Run `main.py` in PyCharm, or execute:

```bash
python main.py
```

## 4. Correct order of use

### Step 1: Register students

1. Click **Register Student**.
2. Enter the student ID, full name and department.
3. Click **Start Capture**.
4. Look at the camera.
5. Move your head slightly left and right while samples are captured.
6. Keep only one face visible during registration.

Press `Q` to cancel the capture.

### Step 2: Train the model

Click **Train Model** after registration.

Train again whenever you add another student. If you skip retraining, the new
student cannot be recognized.

### Step 3: Start attendance

Click **Start Attendance**.

A green rectangle indicates a recognized face. Attendance is written after the
same identity is recognized across multiple consecutive frames.

Press `Q` to close the attendance camera.

### Step 4: Export records

Click **Export CSV** to save all attendance records.

## Command-line use

### Register a student

```bash
python register_student.py --id "MCA001" --name "Raghav Sharma" --department "MCA"
```

Use a different camera:

```bash
python register_student.py --id "MCA001" --name "Raghav Sharma" --camera 1
```

### Train the model

```bash
python train_model.py
```

### Take attendance

```bash
python take_attendance.py
```

Different camera:

```bash
python take_attendance.py --camera 1
```

### Print database records

```bash
python view_database.py
```

### Delete all project data

```bash
python reset_project_data.py
```

You must type `DELETE` when asked.

## Project structure

```text
face_recognition_attendance_opencv/
├── main.py
├── config.py
├── database.py
├── vision.py
├── register_student.py
├── train_model.py
├── take_attendance.py
├── view_database.py
├── reset_project_data.py
├── requirements.txt
├── README.md
├── data/
│   ├── attendance.db
│   └── faces/
├── models/
└── exports/
```

`attendance.db` is created automatically the first time the application runs.

## Changing recognition sensitivity

Open `config.py` and adjust:

```python
LBPH_DISTANCE_THRESHOLD = 68.0
```

LBPH distance is lower for a closer match.

- Too many unknown faces are accepted: reduce the value, for example to `58.0`.
- Registered faces are often rejected: increase it carefully, for example to
  `72.0`.

Do not blindly increase it. A high threshold creates more false matches.

## Camera does not open

- Close Camera, OBS, Discord, browser camera tabs and other webcam applications.
- Change the Camera value in the GUI from `0` to `1`.
- Confirm camera permission is enabled in Windows settings.
- Try the terminal command:

```bash
python take_attendance.py --camera 1
```

## Database rule

The attendance table has a unique rule for:

```text
student + attendance date
```

Therefore, recognizing the same person repeatedly on the same day does not
create duplicate records.
