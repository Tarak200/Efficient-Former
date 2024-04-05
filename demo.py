import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Efficientformer_model = torch.load('new_model/model.pth')
Efficientformer_model.to(device)

st.set_page_config(page_title="EfficientFormer")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("EfficientFormer Image Classifier")

from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
mobileNet_model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")



uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    input_image = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        start_time = time.time()
        output = Efficientformer_model(input_image)
        end_time = time.time()

    total_time = (end_time - start_time) * 1000
    
    label = torch.argmax(output).item()

    if label == 0:
        st.success("Prediction: Fish")
    elif label == 1:
        st.success("Prediction: Salamander")
    else:
        st.success("Prediction: Frog")

    st.write(f"Time taken for inference: {total_time:.2f} ms")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        start_time = time.time()
        logits = mobileNet_model(**inputs)
        end_time = time.time()
    total_time = (end_time - start_time) * 1000
    logits = logits.logits
    predicted_label = logits.argmax(-1).item()
    st.write(f"Time taken for inference for mobielNet: {total_time:.2f} ms")
    print(mobileNet_model.config.id2label[predicted_label])


