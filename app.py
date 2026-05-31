import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    data = joblib.load('crop_model_production.pkl')
    return data['model'], data['num_encod']

model, scaler = load_model()

# ── Crop info dict ──────────────────────────────────────────────────────────────
crop_info = {
    'rice':        ('🌾', 'Rice grows best in warm, humid conditions with high rainfall.'),
    'maize':       ('🌽', 'Maize thrives in moderate temperature with medium rainfall.'),
    'chickpea':    ('🫘', 'Chickpea prefers cool, dry climate with low humidity.'),
    'kidneybeans': ('🫘', 'Kidney beans need moderate rainfall and cool temperature.'),
    'pigeonpeas':  ('🌿', 'Pigeon peas grow in semi-arid tropical conditions.'),
    'mothbeans':   ('🌱', 'Moth beans are drought-resistant, ideal for arid regions.'),
    'mungbean':    ('🌱', 'Mung beans prefer warm, humid weather with moderate rain.'),
    'blackgram':   ('🌿', 'Black gram thrives in warm humid tropical conditions.'),
    'lentil':      ('🫘', 'Lentils prefer cool, dry climate — excellent protein source.'),
    'pomegranate': ('🍎', 'Pomegranate loves dry climate with high humidity.'),
    'banana':      ('🍌', 'Banana thrives in tropical conditions with high nutrients.'),
    'mango':       ('🥭', 'Mango needs hot temperatures and moderate rainfall.'),
    'grapes':      ('🍇', 'Grapes grow best in cool temperature with moderate humidity.'),
    'watermelon':  ('🍉', 'Watermelon loves hot, dry climate — needs low rainfall.'),
    'muskmelon':   ('🍈', 'Muskmelon thrives in very warm weather with low rainfall.'),
    'apple':       ('🍎', 'Apple requires cold winter, moderate summer, high humidity.'),
    'orange':      ('🍊', 'Orange grows in subtropical climate with moderate rainfall.'),
    'papaya':      ('🍑', 'Papaya loves tropical warmth and high humidity.'),
    'coconut':     ('🥥', 'Coconut thrives in coastal tropical climates.'),
    'cotton':      ('🌸', 'Cotton needs high N, warm temp, and moderate rainfall.'),
    'jute':        ('🌿', 'Jute prefers warm, humid climate with high rainfall.'),
    'coffee':      ('☕', 'Coffee grows best in warm temperatures with moderate rain.'),
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌾 Crop Recommendation System")
st.markdown("#### Enter your **soil and climate data** below to find out which crop is best suited for your land.")
st.markdown("---")

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("🧪 Soil Nutrient Levels")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0,
                        help="Nitrogen content in soil (kg/ha)")

with col2:
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0,
                        help="Phosphorus content in soil (kg/ha)")

with col3:
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=210.0, value=50.0, step=1.0,
                        help="Potassium content in soil (kg/ha)")

st.markdown("")
st.subheader("🌤️ Climate Conditions")
col4, col5 = st.columns(2)

with col4:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity    = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)

with col5:
    ph       = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1,
                                help="pH of soil (0=acidic, 14=alkaline, 7=neutral)")
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=100.0, step=1.0)

st.markdown("---")

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🌱 Recommend Crop", use_container_width=True):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                               columns=['N','P','K','temperature','humidity','ph','rainfall'])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]

    emoji, description = crop_info.get(prediction, ('🌱', 'A great crop for your conditions.'))

    st.markdown("---")
    st.success(f"### {emoji} Recommended Crop: **{prediction.upper()}**")
    st.info(f"💡 {description}")

    st.markdown("#### 📊 Your Input Summary")
    summary = pd.DataFrame({
        'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)',
                      'Temperature', 'Humidity', 'pH', 'Rainfall'],
        'Value': [f"{N} kg/ha", f"{P} kg/ha", f"{K} kg/ha",
                  f"{temperature} °C", f"{humidity} %", f"{ph}", f"{rainfall} mm"]
    })
    st.table(summary)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("ML Project | Crop Recommendation System | KNN Classifier | scikit-learn")
