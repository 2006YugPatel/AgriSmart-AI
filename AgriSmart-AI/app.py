import streamlit as st
import joblib
import pandas as pd
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AgriSmart AI",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("crop_model.pkl")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #F5FFF5;
}
h1 {
    color: #2E7D32;
}
.stButton>button {
    background-color: #2E7D32;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size:18px;
}
.stButton>button:hover{
    background-color:#1B5E20;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.title("🌱 AgriSmart AI")
st.subheader("AI-Based Smart Crop Recommendation System")

st.info("Enter your soil and weather details. The AI model will recommend the most suitable crop.")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("🌾 Input Values")

N = st.sidebar.number_input("Nitrogen (N)", 0, 150, 90)
P = st.sidebar.number_input("Phosphorus (P)", 0, 150, 40)
K = st.sidebar.number_input("Potassium (K)", 0, 150, 40)

temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 200.0)

# -------------------------------
# Data
# -------------------------------
data = pd.DataFrame(
    [[N, P, K, temperature, humidity, ph, rainfall]],
    columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ],
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🌾 Recommend Crop"):

    prediction = model.predict(data)
    crop = prediction[0]

    st.success(f"✅ Recommended Crop: **{crop.title()}**")

    image_path = os.path.join("images", f"{crop}.jpg")

    if os.path.exists(image_path):
        st.image(image_path, width=450)

    st.markdown("## 🌱 Crop Information")

    crop_info = {
        "rice": "Rice grows best in warm climates with plenty of water.",
        "wheat": "Wheat prefers cool weather and fertile soil.",
        "maize": "Maize requires good sunlight and moderate rainfall.",
        "cotton": "Cotton grows well in black soil with warm temperatures.",
        "banana": "Banana needs high humidity and rich fertile soil.",
        "apple": "Apple grows in cool hilly regions.",
        "mango": "Mango prefers tropical climates with well-drained soil.",
        "grapes": "Grapes require sunny weather and moderate irrigation.",
        "orange": "Orange grows well in slightly acidic soil.",
        "coffee": "Coffee grows best in cool climates with shade."
    }

    if crop in crop_info:
        st.write(crop_info[crop])
    else:
        st.write("Suitable crop predicted by the AI model.")

    st.markdown("### 💡 Farming Tips")

    st.success("""
✔ Use good quality seeds.

✔ Test your soil before planting.

✔ Apply fertilizers as recommended.

✔ Irrigate regularly.

✔ Monitor pests and diseases.
""")

    st.balloons()

# -------------------------------
# Model Information
# -------------------------------
st.markdown("---")

st.subheader("📊 Model Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "2200")

with col2:
    st.metric("Features", "7")

with col3:
    st.metric("Algorithm", "Random Forest")

# -------------------------------
# SDG
# -------------------------------
st.markdown("---")

st.subheader("🌍 Sustainable Development Goals")

st.write("""
✅ SDG 2 – Zero Hunger

✅ SDG 15 – Life on Land

This AI system helps farmers select suitable crops based on soil nutrients and weather conditions, supporting sustainable agriculture.
""")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")

st.caption(
    "| GTU Skills4Future Internship | AgriSmart AI Capstone Project"
)