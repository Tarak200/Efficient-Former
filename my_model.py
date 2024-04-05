from transformers import AutoImageProcessor, EfficientFormerForImageClassificationWithTeacher
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
model = EfficientFormerForImageClassificationWithTeacher.from_pretrained("snap-research/efficientformer-l1-300")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])