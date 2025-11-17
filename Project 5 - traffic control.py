# To explore and preprocess this dataset for object detection, the following code will cover:
#Data Loading: Reading the image and annotation data.
#Basic Exploration: Visualizing images and class distributions.
#Preprocessing: Applying common image transformations and checking the annotations.
#Data Augmentation: Creating additional variations to enhance the model’s performance.
#This code assumes you have organized the dataset locally with images and annotations in a YOLO-compatible structure.

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
import numpy as np
from PIL import Image

# Define the path for images and annotations
image_dir = r'D:\SBC_PROIECT\road_detection\train\images'
annotation_dir = r'D:\SBC_PROIECT\road_detection\train\labels'

# Classes present in the dataset
classes = ["Car", "Different-Traffic-Sign", "Green-Traffic-Light", "Motorcycle", 
           "Pedestrian", "Pedestrian-Crossing", "Prohibition-Sign", 
           "Red-Traffic-Light", "Speed-Limit-Sign", "Truck", "Warning-Sign"]

# Load image and annotation paths
image_paths = glob(os.path.join(image_dir, '*.jpg'))  # Adjust extension if different
annotation_paths = glob(os.path.join(annotation_dir, '*.txt'))

# Data Exploration
print(f"Total Images: {len(image_paths)}")
print(f"Total Annotations: {len(annotation_paths)}")

# Sample visualization
def visualize_sample_images(image_paths, n=5):
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(image_paths[:n]):
        img = Image.open(img_path)
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

visualize_sample_images(image_paths)

# Class distribution
def get_class_distribution(annotation_paths, classes):
    class_counts = {cls: 0 for cls in classes}
    for annotation_path in annotation_paths:
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_name = classes[class_id]
                class_counts[class_name] += 1
    return class_counts

class_distribution = get_class_distribution(annotation_paths, classes)
class_df = pd.DataFrame(list(class_distribution.items()), columns=['Class', 'Count'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Count', data=class_df)
plt.xticks(rotation=90)
plt.title("Class Distribution in Dataset")
plt.show()

# Data Preprocessing
def preprocess_images(image_paths, target_size=(640, 640)):
    processed_images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        processed_images.append(img)
    return processed_images

processed_images = preprocess_images(image_paths[:5])  # Display first few processed images

# Visualize preprocessed images
plt.figure(figsize=(15, 10))
for i, img in enumerate(processed_images):
    plt.subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.show()

# Data Augmentation (example of brightness adjustment)
def augment_brightness(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

# Visualize augmented images
augmented_images = [augment_brightness(img, factor=1.2) for img in processed_images]

plt.figure(figsize=(15, 10))
for i, img in enumerate(augmented_images):
    plt.subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.show()

# Annotation parsing and checking sample coordinates
def load_yolo_annotations(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes

# Visualize sample image with bounding boxes
def plot_boxes(img_path, annotation_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    boxes = load_yolo_annotations(annotation_path)

    for box in boxes:
        class_id, x_center, y_center, box_width, box_height = box
        x1 = int((x_center - box_width / 2) * w)
        y1 = int((y_center - box_height / 2) * h)
        x2 = int((x_center + box_width / 2) * w)
        y2 = int((y_center + box_height / 2) * h)
        
        class_name = classes[int(class_id)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Plot a sample image with bounding boxes
plot_boxes(image_paths[0], annotation_paths[0])