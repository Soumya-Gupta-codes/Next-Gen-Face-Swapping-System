# Next-Gen Face Swapping System

This repository implements the Next-Gen Face Swapping System using OpenCV and dlib (Delaunay triangulation, affine warping, and seamless cloning).

---

## Requirements & Setup

1. **Python 3.10 - 3.12**
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Windows (CMD):
   .venv\Scripts\activate.bat
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the 68-face-landmarks model file**:
   ```bash
   python download_model.py
   ```
   This downloads and decompresses `shape_predictor_68_face_landmarks.dat` into `models/`.

---

## How to Run

### 1. Real-Time Webcam Face Swapping
Run with default webcam (camera 0) and default sample face:
```bash
python real_time_face_swapping_by_turners.py
```
Press `q` in any OpenCV window to exit.

### 2. Using a Custom Source Face
Swap with your own reference face:
```bash
python real_time_face_swapping_by_turners.py --source path/to/your_face.jpg
```

### 3. Static Image-to-Image Swapping (No Webcam Required)
```bash
python real_time_face_swapping_by_turners.py --source assets/source_face.jpg --target assets/target_face.jpg
```

### 4. Headless Execution
Process a frame and save the result to file without opening GUI windows:
```bash
python real_time_face_swapping_by_turners.py --source assets/source_face.jpg --target assets/target_face.jpg --headless
```
The output image will be saved to `assets/output_swapped.jpg`.

---

## Command-Line Options

| Argument | Description | Default |
|---|---|---|
| `--source` | Path to source face image | `assets/source_face.jpg` |
| `--target` | Path to target image or video file (if omitted, uses webcam) | `None` (webcam) |
| `--camera` | Webcam device index | `0` |
| `--model` | Path to shape predictor `.dat` model | `models/shape_predictor_68_face_landmarks.dat` |
| `--headless` | Run in headless mode (no GUI window) | `False` |
| `--output` | Destination path for saved output image | `assets/output_swapped.jpg` |

---

## Abstract
Face swapping has been a thriving genre of work which primarily is associated with the replacement or substitution of one referral face on the face of the other person, be it by means of still images or in real time. This paper exhibits our research on the Real-Time Face Swapping algorithm, using the reference image of the user and henceforth considering it as our input image instead of using a dataset and working on a live model. The input image facial features and attributes are extracted and are replaced to make the final resulting output through our model. We start by training our model to learn the facial alignments and features to be able to recognize a face and extract its features before swapping it. Now, we head over to the process of image warping by the dissociation and partitioning of the user’s face and its background. To increase the precision of the model, in the second part of parsing the face is performed to cancel out other features of the live image, hence our output is obtained.
