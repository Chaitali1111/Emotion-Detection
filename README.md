Emotion Detection using Deep Learning (CNN) | OpenCV + TensorFlow + Streamlit

📌 Project Overview

This project is an End-to-End Facial Emotion Detection System that classifies human emotions from images using a Convolutional Neural Network (CNN).
The model is trained on the FER2013 dataset and deployed using Streamlit as a web app.

Users can upload an image, and the app will: ✔ Detect the face
✔ Process it
✔ Predict the emotion
✔ Display the result in real-time


---

🚀 Features

🎯 Emotion classification (Happy, Sad, Angry, Neutral, Fear, Disgust, Surprise)

🧠 Custom CNN built using TensorFlow/Keras

🖼 Face detection using OpenCV Haarcascade

🌐 Clean & interactive Streamlit UI

📁 Well-organized folder structure (Industry standard)

🔍 Real-time prediction for uploaded images

🧪 Model trained on ~35K+ images



---

📂 Project Structure

Emotion-Detection/
│── app/
│   ├── model/
│   │   ├── emotion_cnn_model.h5
│   │   ├── labels.txt
│   ├── streamlit_app.py    ← Streamlit UI
│   ├── haarcascade_frontalface_default.xml
│
│── data/
│   ├── train/
│   ├── test/
│   ├── validation/
│
│── training.py             ← CNN model training script
│── requirements.txt
│── README.md


---

🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

Conv2D → ReLU

MaxPooling

Conv2D → ReLU

MaxPooling

Flatten

Dense (128 units)

Dense (7 output classes with Softmax)


Loss: categorical_crossentropy
Optimizer: Adam


---

💾 Dataset

Dataset used: FER-2013

48x48 grayscale facial images

~35,000 labeled images

7 emotion categories


Dataset Folder Structure:

data/train/<emotion>/
data/test/<emotion>/
data/validation/<emotion>/


---

▶️ How to Run Locally

1️⃣ Create and Activate Virtual Environment

python -m venv .venv
.venv\Scripts\activate     # Windows

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Train the Model (Optional)

python training.py

4️⃣ Run the Streamlit App

streamlit run app/streamlit_app.py

App will open at:
👉 http://localhost:8501


---

🎨 Streamlit UI Preview

Users can:

Upload an image

Detect face

See predicted emotion instantly



---

📊 Labels

Stored in:
app/model/labels.txt

Example:

0 Angry
1 Disgust
2 Fear
3 Happy
4 Neutral
5 Sad
6 Surprise


---

🛠 Technologies Used

Technology	Purpose

TensorFlow	Model training (CNN)
Keras	Deep learning framework
OpenCV	Face detection
Streamlit	Web interface
NumPy / Pandas	Data handling
FER2013 Dataset	Training data



---

💡 Future Enhancements

Real-time webcam emotion detection

Deployment on Render / HuggingFace Spaces

Improve model accuracy using transfer learning

Add age & gender detection

----

📎 Author
Chaitali K.
