import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
# load a model pre-trained on COCO

import PIL 
from PIL import Image
import os
def load_yolo_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model

def detect_objects_yolo(model, image_path):
    image = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction

def get_top_objects(prediction, top_k=5):
    boxes = prediction[0]['boxes'][:top_k]
    labels = prediction[0]['labels'][:top_k]

    # Replace these labels with the actual class labels based on your dataset
    class_labels = ["person", "car", "cycle", "skateboard"]

    objects = []
    for label, box in zip(labels, boxes):
        if label < len(class_labels):  # Check if the label is within the range of class_labels
            objects.append({"label": class_labels[label], "box": box.tolist()})
        else:
            objects.append({"label": f"Unknown label ({label})", "box": box.tolist()})

    return objects

from tqdm import tqdm  # Import tqdm for progress bar

def process_images(folder_path):
    model = load_yolo_model()
    images_objects = {}

    # Use tqdm to display a progress bar
    for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            prediction = detect_objects_yolo(model, image_path)
            objects = get_top_objects(prediction)
            images_objects[filename] = objects
    with open("op.json", 'w') as f:
        json.dump(images_objects, f)
    return images_objects

# Example usage:
folder_path = "predictions2"
result = process_images(folder_path)
print(result)

