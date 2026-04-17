import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle

# -------- LOAD MODEL --------
model = load_model("food_model.h5")

# -------- LOAD CSV --------
df = pd.read_csv("nutrition.csv")

# -------- LOAD CLASS NAMES --------
class_indices = pickle.load(open("class_indices.pkl", "rb"))
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

# -------- UI --------
st.set_page_config(page_title="Food Analyzer", layout="centered")
st.title("Food Recognition & Nutrition Analyzer")

uploaded_file = st.file_uploader("Upload Food Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img.save("temp.jpg")

    # -------- STEP 1: CLASSIFICATION --------
    img_resized = image.load_img("temp.jpg", target_size=(224,224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    food = class_names[predicted_index]

    # -------- STEP 2: BASIC COUNT (OpenCV) --------
    img_cv = cv2.imread("temp.jpg")
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    auto_count = sum(1 for c in contours if cv2.contourArea(c) > 500)

    if auto_count == 0:
        auto_count = 1


    # -------- STEP 3: NUTRITION --------
    row = df[df['food'] == food]

    st.subheader("Detection Result")
    st.write(f"**Food:** {food}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # -------- USER INPUT --------
    st.subheader("Adjust Quantity")

    st.write(f"Estimated Count: **{auto_count}**")

    count = st.slider("Select correct count", 1, 20, auto_count)

    if not row.empty:
        calories = row['calories'].values[0] * count
        protein = row['protein'].values[0] * count
        fat = row['fat'].values[0] * count
        carbs = row['carbs'].values[0] * count
        sugar = row['sugar'].values[0] * count

        low = calories * 0.9
        high = calories * 1.1

        st.subheader("Nutrition Analysis")

        st.write(f"**Calories:** {low:.1f} - {high:.1f}")
        st.write(f"**Protein:** {protein:.1f} g")
        st.write(f"**Fat:** {fat:.1f} g")
        st.write(f"**Carbohydrates:** {carbs:.1f} g")
        st.write(f"**Sugar:** {sugar:.1f} g")

    else:
        st.warning("Nutrition data not found.")

