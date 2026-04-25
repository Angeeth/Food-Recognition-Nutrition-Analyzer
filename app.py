import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle

# -------- LOAD MODELS --------
model1 = load_model("food_model.h5")
model2 = load_model("food_model_2.h5")

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

    # -------- STEP 1: PREPROCESS --------
    img_resized = image.load_img("temp.jpg", target_size=(224,224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- STEP 2: DUAL MODEL PREDICTION --------
    with st.spinner("Analyzing image..."):
        pred1 = model1.predict(img_array)
        pred2 = model2.predict(img_array)

    # Model 1
    index1 = np.argmax(pred1)
    conf1 = float(np.max(pred1))
    food1 = class_names[index1]

    # Model 2
    index2 = np.argmax(pred2)
    conf2 = float(np.max(pred2))
    food2 = class_names[index2]

    # -------- SELECT BEST MODEL --------
    if conf1 > conf2:
        food = food1
        confidence = conf1
        selected_model = "Model 1 (food_model.h5)"
    else:
        food = food2
        confidence = conf2
        selected_model = "Model 2 (food_model_2.h5)"

    # -------- STEP 3: BASIC COUNT (OpenCV) --------
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

    # -------- STEP 4: NUTRITION --------
    row = df[df['food'] == food]

    st.subheader("Detection Result")
    st.write(f"**Food:** {food}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
#    st.write(f"**Selected Model:** {selected_model}")

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