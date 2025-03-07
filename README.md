# Traffic Road Object Detection Using YOLOv8

## Overview
This project focuses on detecting traffic-related objects using the YOLOv8 model. The dataset contains images with labeled objects such as vehicles, pedestrians, and road signs. The model is trained using Ultralytics' YOLO framework to achieve real-time object detection.

## Dataset
- **Source**: Kaggle Dataset - `traffic-road-object-detection-dataset-using-yolo`
- **Structure**:
  - `train/` - Training images
  - `val/` - Validation images
  - `classes.txt` - List of class names

## Setup and Installation
### 1. Install Dependencies
```sh
pip install ultralytics --quiet
```

### 2. Define Dataset Configuration
The `dataset.yaml` file is generated with paths and class mappings.

## Training the YOLOv8 Model
```python
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="/kaggle/working/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolov8_training",
    device=0
)
```

## Validation
After training, validate the model to assess its performance:
```python
results = model.val()
```

## Inference
Run inference on validation images to generate predictions:
```python
results = model.predict(
    source="/kaggle/input/traffic-road-object-detection-dataset-using-yolo/val/images",
    save=True,
    save_dir="/kaggle/working/runs/predict",
    conf=0.5
)
```

## Visualizing Predictions
To display the predicted images:
```python
import cv2
import matplotlib.pyplot as plt
import os

predicted_dir = "/kaggle/working/runs/detect/predict2"

if os.path.exists(predicted_dir):
    predicted_images = [os.path.join(predicted_dir, img) for img in os.listdir(predicted_dir) if img.endswith(".jpg")]
    for img_path in predicted_images[:5]:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
```

## Generating a Video from Predictions
```python
import cv2
import os

def images_to_video(image_folder, output_file, frame_rate=30):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))])
    if not images:
        print("No images found.")
        return
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved at {output_file}")

output_video = "/kaggle/working/predicted_video.mp4"
images_to_video(predicted_dir, output_video, 30)
```

## Results
- The trained YOLOv8 model can successfully detect traffic-related objects in real-world scenarios.
- Predicted bounding boxes and class labels can be visualized.
- A video summarizing the predictions has been generated.

## Future Work
- Improve detection accuracy by training on a larger dataset.
- Experiment with different YOLO model sizes (e.g., `yolov8s.pt`, `yolov8m.pt`).
- Fine-tune hyperparameters for better performance.

## Acknowledgments
- Ultralytics for the YOLO framework.
- Kaggle for providing the dataset.


