# Face Detection Project with R and Python

This project uses R with the `reticulate` package to run Python code for real-time face detection, encoding, and recognition.

## Project Structure

```
face_detection_project/
├── face_detection.R           # Main R script with logic
├── requirements.txt           # Python dependencies
└── python_modules/
    └── face_detector.py       # Python face detection module
```

## Setup Instructions

### 1. Install R Packages

Open R/RStudio and run:
```r
install.packages("reticulate")
```

### 2. Install Python Dependencies

**Option A: Using pip**
```bash
pip install -r requirements.txt
```

**Option B: In R using reticulate**
```r
library(reticulate)
py_install(c("opencv-python", "face-recognition", "numpy", "cmake", "dlib"), 
           pip = TRUE)
```

### 3. Configure reticulate to use the correct Python environment

In your R script or console:
```r
library(reticulate)

# Point to your Python executable
# Windows example:
use_python("C:\\Users\\YourUsername\\AppData\\Local\\Programs\\Python\\Python311\\python.exe")

# Or use a virtual environment
# use_virtualenv("path/to/venv")
```

## Usage

### Real-time Face Detection (Basic)

```r
source("face_detection.R")
run_face_detection()
```

This will open your camera and display detected faces in real-time.

### Face Detection with Known Faces

1. Create a directory structure like:
```
known_faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
```

2. Run in R:
```r
source("face_detection.R")
run_face_detection(known_faces_dir = "known_faces")
```

### Encode a Single Face from an Image

```r
source("face_detection.R")
face_encoding <- encode_face("path/to/image.jpg")
```

## Features

- **Real-time Face Detection**: Uses OpenCV for camera access and face_recognition library for detection
- **Face Encoding**: Converts faces to 128-dimensional vectors for comparison
- **Face Recognition**: Matches detected faces against known faces
- **Two Detection Models**:
  - HOG (Histogram of Oriented Gradients): Fast, works on CPU
  - CNN (Convolutional Neural Network): More accurate, requires GPU for real-time performance

## Controls

- Press **'q'** to quit the detection loop

## Notes

- The CNN model is more accurate but slower. Use `use_cnn = TRUE` for better accuracy if your computer has a GPU
- By default, every other frame is processed for performance
- Face encodings are 128-dimensional vectors that can be saved and compared later
- The tolerance parameter (default 0.6) controls how strict face matching is

## Troubleshooting

**"Cannot open camera"**: Make sure your camera is connected and not in use by another application

**dlib installation issues on Windows**: You may need to install Visual Studio Build Tools or use a pre-built wheel

**Python not found**: Set the Python path explicitly using `use_python()` or create a virtual environment

## Python Module Reference

The `FaceDetector` class provides these main methods:

- `open_camera()`: Initialize camera
- `close_camera()`: Release camera
- `capture_frame()`: Get a single frame
- `detect_faces_in_frame()`: Detect and encode faces in a frame
- `encode_face_from_image()`: Encode a face from a file
- `compare_faces()`: Compare encodings
- `load_known_faces()`: Load known face encodings
- `display_detection_results()`: Draw results on frame
