# YOLOv5-Based-Deep-Learning-Framework-for-GPR-Hyperbola-Recognition-and-Analysis
This project uses YOLOv5 to detect hyperbolas in Ground Penetrating Radar (GPR) images, enabling automated identification of buried objects. The pipeline includes data preprocessing, training, evaluation, and visualization for efficient subsurface analysis.
## 1. Dataset Description
Used 241 stimulated Ground Penetrating Radar (GPR) images.

Captures subsurface reflections with hyperbolic patterns.

The dataset is not publicly disclosed at this stage.
<img width="1058" height="610" alt="image" src="https://github.com/user-attachments/assets/db3ad935-8950-40b5-a569-5a60d8150a65" />

## 2. Preprocessing: Image Cropping
All GPR images were cropped to remove irrelevant background areas.

This step helps the model focus on meaningful regions containing hyperbolic signatures.

Images were saved in .png format with a uniform resolution of 256 × 256.
<img width="942" height="601" alt="image" src="https://github.com/user-attachments/assets/93104cd1-78bb-45bc-a656-38e5f9031002" />

## 3. To improve generalization and increase dataset size, the following augmentations were applied to each original image:

Original

CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE + Horizontal Stretch

CLAHE + Vertical Stretch

CLAHE + Flip (Horizontal)

CLAHE + Rotated

CLAHE + Gaussian Noise

CLAHE + Speckle Noise

CLAHE + Gaussian Blur

✅ Total augmented images = 241 original × 9 variations = 2169 images (all 256 × 256)
<img width="1400" height="242" alt="image" src="https://github.com/user-attachments/assets/eb1cbf67-2dbe-4ac2-bbbd-77550b2bdae7" />
<img width="1402" height="237" alt="image" src="https://github.com/user-attachments/assets/eee4867c-d988-4684-90af-fc5e3a330ade" />
<img width="1410" height="245" alt="image" src="https://github.com/user-attachments/assets/52455f69-63e3-41c0-9205-7c93f7219603" />

This step simulates real-world GPR variations (e.g., distortion, noise) and ensures the model learns robust features.
## 4. Manual Annotation Using MATLAB Image Labeler
Used MATLAB Image Labeler for manual bounding box annotations.
<img width="1915" height="990" alt="matlab 1" src="https://github.com/user-attachments/assets/188791b0-5b4c-4ffb-bda8-659bde30894e" />
<img width="1911" height="1007" alt="matlab 2" src="https://github.com/user-attachments/assets/ed587f1e-7d1d-40f5-89a9-4595a4dc25a5" />

Each hyperbola, corresponding to a subsurface object, was manually labeled with a bounding box and class name "hyperbola".

Why? Automated detection requires ground truth bounding boxes to learn the location and shape of hyperbolas.

Alternative Tool: Open command prompt in local: labelImg

pip install labelImg
labelImg
Opens a GUI to load the augmented image folder and draw bounding boxes.

For each box, enter label: hyperbola

## 5. Saving and Converting Annotations
Annotations were saved in Pascal VOC .xml format (default for labelImg).

Converted these .xml files into:

-> .csv format for use with RetinaNet.

-> .txt format (YOLO format: class x_center y_center width height) for use with YOLOv5. This becomes the labels of each hyperbolas detected.
The.txt file shouls look like this
<img width="440" height="292" alt="image" src="https://github.com/user-attachments/assets/04c70a5d-21aa-4ad0-8b49-db1c1c03778d" />

This ensures compatibility with the chosen object detection framework.
## 6. Hyperbola Detection using YOLOv5
✅ Why Use YOLOv5?
YOLO (You Only Look Once) is a real-time object detection model known for its:

Speed: YOLOv5 is extremely fast and suitable for high-throughput tasks like scanning large GPR datasets.

Accuracy: It delivers a good balance between speed and precision, especially with hyperbolic shapes in GPR.

Simplicity: Easy to train, customize, and deploy using PyTorch.

Lightweight: Compared to older models like Faster R-CNN or RetinaNet, YOLOv5 requires less compute power and is easier to optimize.

Significance of YOLOv5 in This Project
Detects small, curved patterns (hyperbolas) in cluttered GPR signals.

Works well with 256×256 images — preserving spatial details of hyperbolas.

Supports custom datasets and small object detection — ideal for subsurface signal interpretation.

Helps with automated subsurface utility detection, reducing manual inspection effort.

YOLOv5 Variants (from smallest to largest)
yolov5n – Nano (fastest, lightest, less accurate)

yolov5s – Small (used in this project, ideal for limited datasets like GPR)

yolov5m – Medium

yolov5l – Large

yolov5x – Extra Large (most accurate, but heavier)
Dataset Structure for YOLOv5
To train YOLOv5, the dataset must follow this specific format:

kotlin
Copy
Edit
yolo_dataset.zip
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   └── val
└── gpr_data.yaml  ← describes class names and dataset paths
### Train-Test Split (80:20)
To prepare:

Randomly split your 2000 augmented images into:

1600 training images + labels (80%)

400 validation images + labels (20%)

Ensure that:

Each image has a corresponding .txt label file in YOLO format.

All label files follow the syntax:
class x_center y_center width height (all values normalized between 0 and 1)

Save them into appropriate folders:

/images/train, /images/val

/labels/train, /labels/val

Create a gpr_data.yaml file like this:
path: /content/GPR-YOLO-Dataset/yolo_dataset
train: images/train
val: images/val

nc: 1
names: ['hyperbola']






