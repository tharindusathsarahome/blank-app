# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os # Still needed for get_default_device

# --- 1. Define Model Architecture (Must be identical to training) ---
class ImageClassificationBase(nn.Module):
    # (Keep the class definition exactly as in your original script)
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        # accuracy function would need to be defined if used here or remove this line for pure inference
        # acc = accuracy(out, labels)
        # return {'val_loss': loss.detach(), 'val_acc': acc}
        return {'val_loss': loss.detach()} # Simplified for inference

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        # batch_accs = [x['val_acc'] for x in outputs]
        # epoch_acc = torch.stack(batch_accs).mean()
        # return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        return {'val_loss': epoch_loss.item()} # Simplified for inference

    def epoch_end(self, epoch, result):
        # This method is mostly for training, can be simplified or removed for inference app
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result.get('train_loss', float('nan')), result['val_loss'], result.get('val_acc', float('nan'))))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# --- 2. Define Helper functions for device ---
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# --- 3. Class names (Must match the order and names from training) ---
kept_classes = [
    'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight',
    'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Tomato___Target_Spot',
    'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Corn_(maize)___Common_rust_',
    'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Corn_(maize)___healthy'
]
# IMPORTANT: Ensure this matches the `train.classes` from your notebook.
# ImageFolder sorts class names (folder names) alphabetically.
CLASS_NAMES = sorted(kept_classes)
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = 'plant-disease-model.pth' # Ensure this file is in the same directory

# --- 4. Load Model (Cached for performance) ---
@st.cache_resource # Use cache_resource for models and other non-data objects
def load_pytorch_model(model_path, num_classes):
    device = get_default_device()
    model = CNN_NeuralNet(in_channels=3, num_diseases=num_classes)
    try:
        # Load on CPU if GPU not available, or if model was saved on GPU and running on CPU
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.success(f"Model loaded on {device}.")
    except FileNotFoundError:
        st.sidebar.error(f"Error: Model file '{model_path}' not found.")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None
    model = to_device(model, device)
    model.eval() # Set model to evaluation mode
    return model, device

# --- 5. Preprocess Image ---
def preprocess_image(image_pil):
    """Loads a PIL image and applies the necessary transformations."""
    # Define the same transformations used during training/validation
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Match training image size
        transforms.ToTensor()          # Converts to tensor, scales to [0,1], and changes to CxHxW
    ])
    return transform(image_pil)

# --- 6. Predict Function ---
def predict(image_tensor, model_instance, device, class_names_list):
    """Converts image tensor to batch, predicts, and returns class name and confidence."""
    if image_tensor is None:
        return "Error in preprocessing", 0.0

    # Add a batch dimension (CxHxW -> 1xCxHxW)
    input_batch = image_tensor.unsqueeze(0)
    input_batch = to_device(input_batch, device)

    with torch.no_grad(): # No need to track gradients for inference
        output = model_instance(input_batch)

    probabilities = F.softmax(output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class_name = class_names_list[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    return predicted_class_name, confidence_score

# --- 7. Streamlit App UI ---
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

st.title("🌿 Plant Crop Disease Detection")
st.markdown("""
Upload an image of a plant leaf (corn, potato, or tomato) to detect potential diseases.
This model can identify the following conditions:
""")
st.markdown(f"<small>{', '.join(CLASS_NAMES)}</small>", unsafe_allow_html=True)
st.markdown("---")

# Load Model (this will run once and be cached)
model, device = load_pytorch_model(MODEL_PATH, NUM_CLASSES)

if model is None:
    st.error("Model could not be loaded. Please check the logs and ensure the model file is present.")
else:
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        pil_image = Image.open(uploaded_file).convert("RGB") # Ensure image is RGB
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.write("") # Spacer
            if st.button("🔍 Classify Image"):
                with st.spinner("Analysing the leaf..."):
                    # Preprocess
                    img_tensor = preprocess_image(pil_image)

                    # Predict
                    predicted_class, confidence = predict(img_tensor, model, device, CLASS_NAMES)

                    st.success(f"**Predicted Condition:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence:.2f}%")

                    if "healthy" not in predicted_class.lower() and confidence > 50: # Example threshold
                        st.warning("The plant may require attention.")
                    elif "healthy" in predicted_class.lower():
                        st.balloons()
                        st.success("The plant appears to be healthy!")


st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("This app uses a Convolutional Neural Network (CNN) trained on the New Plant Diseases Dataset to identify plant diseases.")
st.sidebar.markdown("Developed using PyTorch and Streamlit.")