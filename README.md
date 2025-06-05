# Plant Disease Detection App 🌿

A simple web app to detect diseases in plant leaves using a Deep Learning model.

Developed by: Dilanka Kasun

## Features

*   Upload an image of a plant leaf (Corn, Potato, Tomato, and more).
*   Get a prediction of the plant's condition.
*   See the model's confidence score.

## How to Use

1.  **Run the App Locally:**
    *   Clone this repository.
    *   Install requirements: `pip install -r requirements.txt`
    *   Make sure `plant-disease-model.pth` is in the same folder.
    *   Run: `streamlit run app.py`
    *   Open in your browser (usually `http://localhost:8501`).

2.  **Using the App:**
    *   Upload a leaf image.
    *   Click "Classify Image".
    *   View the prediction!

## Model

Uses a pre-trained Convolutional Neural Network (CNN) to identify 38 different plant conditions.

---

This app is for educational and demonstration purposes.