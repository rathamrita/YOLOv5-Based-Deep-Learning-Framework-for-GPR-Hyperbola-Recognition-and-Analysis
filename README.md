# YOLOv5-Based-Deep-Learning-Framework-for-GPR-Hyperbola-Recognition-and-Analysis
This project uses YOLOv5 to detect hyperbolas in Ground Penetrating Radar (GPR) images, enabling automated identification of buried objects. The pipeline includes data preprocessing, training, evaluation, and visualization for efficient subsurface analysis.
## 1. Dataset Description
Used 241 stimulated Ground Penetrating Radar (GPR) images.

Each image is of size 256 × 256 pixels and captures subsurface reflections with hyperbolic patterns.

The dataset is not publicly disclosed at this stage.
## 2. Preprocessing: Image Cropping
All GPR images were cropped to remove irrelevant background areas.

This step helps the model focus on meaningful regions containing hyperbolic signatures.

Images were saved in .png format with a uniform resolution of 256 × 256.
## 3. To improve generalization and increase dataset size, the following augmentations were applied to each original image:

Original

CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE + Horizontal Stretch

CLAHE + Vertical Stretch

CLAHE + Flip (Horizontal)

CLAHE + Gaussian Noise

CLAHE + Speckle Noise

CLAHE + Gaussian Blur

✅ Total augmented images = 241 original × 9 variations = 2169 images (all 256 × 256)

This step simulates real-world GPR variations (e.g., distortion, noise) and ensures the model learns robust features.
## 4. Manual Annotation Using MATLAB Image Labeler
Used MATLAB Image Labeler for manual bounding box annotations.

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

.csv format for use with RetinaNet.

.txt format (YOLO format: class x_center y_center width height) for use with YOLOv5.

This ensures compatibility with the chosen object detection framework.








