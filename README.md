# 🍽️ Food Recognition & Nutrition Analyzer

This project uses Machine Learning to identify food items from images and calculate their nutritional values.

---

## 🚀 Features

- Food image classification using dual CNN models (MobileNetV2 + EfficientNet)
- Confidence-based model selection for improved prediction accuracy
- Object counting using OpenCV
- Nutrition calculation using CSV dataset
- User-adjustable quantity for better accuracy
- Interactive web app using Streamlit

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- Pandas, NumPy

---

## ⚙️ How It Works

1. User uploads a food image
2. Image is processed and passed to two trained models
3. Both models predict the food item
4. The system selects the prediction with higher confidence
5. OpenCV estimates number of items
6. User adjusts quantity if needed
7. Nutrition values are calculated from dataset

---

## 🧠 Model Improvement
- Initially used a single CNN model (MobileNetV2)
- Trained a second model using EfficientNet for better generalization
- Implemented a dual-model inference system
- Final prediction is selected based on highest confidence score, improving robustness

