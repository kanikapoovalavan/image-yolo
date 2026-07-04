# Object Detection using YOLOv8 in Image and Video

This project demonstrates **object detection** in images and videos using the **YOLOv8 (You Only Look Once)** model from the **Ultralytics** library. The model detects objects with high accuracy and displays bounding boxes, class labels, and confidence scores.

## Project Objectives

- Detect objects in images using YOLOv8.
- Detect objects in videos frame by frame.
- Display bounding boxes, object labels, and confidence scores.
- Save the processed output image.

## Tech Stack

- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- PyTorch

## Project Structure

```text
.
## Project Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── proj.py                 # Image object detection
├── provideo.py             # Video object detection
├── goldenretriver.jpg      # Sample input image
├── video_traffic.mp4       # Sample input video
├── output.jpg              # Generated output image
└── output.mp4              # Generated output video
```
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

### Image Object Detection

1. Open `proj.py`.
2. Update the image path if required.
3. Run:

```bash
python proj.py
```

The detected image will be saved as **output.jpg**.

### Video Object Detection

1. Open `provideo.py`.
2. Update the video path if required.
3. Run:

```bash
python provideo.py
```

The video will be processed and displayed or saved depending on the implementation.

## Example Output

The output image contains:

- Bounding boxes around detected objects
- Object class names
- Confidence scores

Example console output:

```text
dog confidence: 0.98
person confidence: 0.95
```

## Requirements

The required Python packages are listed in `requirements.txt`.

## Author

**Kanika S P**  
MSc (Integrated) Information Technology  
College of Engineering, Guindy
