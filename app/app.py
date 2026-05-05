import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Disease Risk Alert",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"], .stApp {
        background-color: #0d0d0d !important;
        color: #e8e8e8 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #2a2a2a !important;
    }

    section[data-testid="stSidebar"] * {
        color: #aaaaaa !important;
    }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 52px;
        color: #f0f0f0;
        line-height: 1.1;
        letter-spacing: -1px;
        margin-bottom: 6px;
    }

    .hero-sub {
        font-size: 15px;
        color: #666666;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, #444 0%, #222 100%);
        margin: 20px 0 32px 0;
    }

    .section-label {
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #555555;
        margin-bottom: 10px;
    }

    .stMultiSelect > div > div {
        background-color: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
        color: #e8e8e8 !important;
    }

    div[data-baseweb="tag"] {
        background-color: #2a2a2a !important;
        border: 1px solid #444 !important;
    }

    div[data-baseweb="tag"] span {
        color: #e8e8e8 !important;
    }

    .stRadio label, .stSlider label {
        color: #aaaaaa !important;
    }

    .card-result {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 26px 22px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .card-result::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }

    .card-disease::before    { background: #00e676; }
    .card-confidence::before { background: #40c4ff; }
    .card-urgency-high::before   { background: #ff5252; }
    .card-urgency-medium::before { background: #ffab40; }
    .card-urgency-low::before    { background: #00e676; }

    .metric-label {
        font-size: 10px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #555555;
        margin-bottom: 10px;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 28px;
        line-height: 1.15;
    }

    .tip-box {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 22px 24px;
    }

    .tip-title {
        font-size: 10px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #555555;
        margin-bottom: 10px;
    }

    .tip-text {
        font-size: 14px;
        color: #cccccc;
        line-height: 1.7;
    }

    .progress-row {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    .progress-name {
        font-size: 13px;
        font-weight: 500;
        color: #dddddd;
        margin-bottom: 8px;
    }

    .progress-track {
        background: #2a2a2a;
        border-radius: 99px;
        height: 5px;
        overflow: hidden;
    }

    .progress-fill {
        height: 5px;
        border-radius: 99px;
        background: linear-gradient(90deg, #40c4ff, #00e676);
    }

    .progress-pct {
        font-size: 11px;
        color: #555555;
        margin-top: 5px;
        text-align: right;
    }

    .urgency-banner {
        border-radius: 10px;
        padding: 14px 20px;
        font-size: 14px;
        margin: 20px 0;
        border-left-width: 3px;
        border-left-style: solid;
    }

    .stButton > button {
        background-color: #e8e8e8 !important;
        color: #0d0d0d !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        height: 3.2em !important;
        width: 100% !important;
        border-radius: 10px !important;
        border: none !important;
    }

    .stButton > button:hover {
        background-color: #ffffff !important;
    }

    .disclaimer {
        font-size: 12px;
        color: #444444;
        text-align: center;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #222222;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model    = joblib.load(os.path.join(BASE_DIR, 'model', 'disease_model.pkl'))
le       = joblib.load(os.path.join(BASE_DIR, 'model', 'label_encoder.pkl'))
symptoms = joblib.load(os.path.join(BASE_DIR, 'model', 'symptom_columns.pkl'))

suggestions = {
    "Diabetes":         {"diet": "Low sugar, high fiber. Avoid white rice & sweets.",                    "lifestyle": "Walk 30 mins daily, check blood sugar weekly.",                "urgency": "High"},
    "Hypertension":     {"diet": "Low salt diet. Eat bananas and leafy greens.",                         "lifestyle": "Reduce stress, limit caffeine, sleep 8 hrs.",                  "urgency": "High"},
    "Anemia":           {"diet": "Iron-rich foods: spinach, lentils, red meat.",                         "lifestyle": "Avoid tea after meals, take iron supplements.",                "urgency": "Medium"},
    "Migraine":         {"diet": "Avoid chocolate, caffeine and processed food.",                        "lifestyle": "Sleep on schedule, avoid bright screens at night.",            "urgency": "Medium"},
    "Fungal infection": {"diet": "Avoid sugar and fermented foods.",                                     "lifestyle": "Keep skin dry, wear breathable clothing.",                     "urgency": "Low"},
    "Allergy":          {"diet": "Avoid known allergens, eat anti-inflammatory foods.",                  "lifestyle": "Keep environment clean, avoid dust and pollen.",               "urgency": "Medium"},
    "Pneumonia":        {"diet": "Warm fluids, vitamin C rich foods.",                                   "lifestyle": "Rest completely, avoid cold air.",                             "urgency": "High"},
    "Dengue":           {"diet": "Papaya leaf juice, coconut water, high fluid intake.",                 "lifestyle": "Complete bed rest, avoid mosquito exposure.",                  "urgency": "High"},
    "Typhoid":          {"diet": "Boiled foods, avoid raw vegetables and street food.",                  "lifestyle": "Complete rest, stay hydrated.",                                "urgency": "High"},
    "Malaria":          {"diet": "Light easily digestible foods, lots of fluids.",                       "lifestyle": "Use mosquito nets, take prescribed medication.",                "urgency": "High"},
    "Acne":             {"diet": "Avoid oily & junk food. Eat zinc-rich foods like pumpkin seeds.",      "lifestyle": "Wash face twice daily, avoid touching face, drink 3L water.",  "urgency": "Low"},
    "Impetigo":         {"diet": "Boost immunity with Vitamin C foods like oranges and amla.",           "lifestyle": "Keep skin clean, avoid sharing towels or clothes.",             "urgency": "Medium"},
    "Jaundice":         {"diet": "Sugarcane juice, lemon water, avoid fatty foods completely.",          "lifestyle": "Complete bed rest, avoid alcohol, drink lots of fluids.",      "urgency": "High"},
    "Chicken pox":      {"diet": "Soft foods, coconut water, avoid spicy food.",                        "lifestyle": "Avoid scratching, stay isolated, keep skin clean.",             "urgency": "High"},
}

urgency_map    = {"High": ("● HIGH", "See a doctor within 2 days"), "Medium": ("● MEDIUM", "Monitor symptoms for 1 week"), "Low": ("● LOW", "Maintain healthy habits")}
urgency_colors = {"High": "#ff5252", "Medium": "#ffab40", "Low": "#00e676"}
urgency_bg     = {"High": "#1a1010", "Medium": "#1a1500", "Low": "#0d1a10"}

st.sidebar.markdown("### Disease Risk Alert")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model**  \nRandom Forest Classifier")
st.sidebar.markdown("**Dataset**  \nKaggle Disease-Symptom")
st.sidebar.markdown("**Coverage**  \n40+ conditions")
st.sidebar.markdown("---")
st.sidebar.markdown("_For informational purposes only. Always consult a qualified doctor._")

st.markdown('<div class="hero-title">Early Disease<br>Risk Alert</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered symptom analysis · Instant health guidance</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="section-label">Symptoms</div>', unsafe_allow_html=True)
    selected = st.multiselect("Select symptoms", options=symptoms, help="Select as many symptoms as apply", label_visibility="collapsed")
    if selected:
        st.markdown(f'<div style="font-size:12px;color:#555;margin-top:6px;">{len(selected)} symptom(s) selected</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-label">Profile</div>', unsafe_allow_html=True)
    age = st.slider("Age", 1, 100, 25, label_visibility="collapsed")
    st.markdown(f'<div style="font-size:12px;color:#555;margin-bottom:8px;">Age: {age}</div>', unsafe_allow_html=True)
    gender = st.radio("Gender", ["Male", "Female", "Other"], label_visibility="collapsed", horizontal=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analyze Risk"):
    if len(selected) == 0:
        st.error("Please select at least one symptom to proceed.")
    else:
        input_vec    = [1 if s in selected else 0 for s in symptoms]
        input_df     = pd.DataFrame([input_vec], columns=symptoms)
        pred_encoded = model.predict(input_df)[0]
        pred_proba   = model.predict_proba(input_df)[0]
        confidence   = round(max(pred_proba) * 100, 2)
        disease      = le.inverse_transform([pred_encoded])[0]

        info = suggestions.get(disease, {"diet": "Eat balanced meals with fruits and vegetables.", "lifestyle": "Stay active, hydrated and sleep 8 hrs daily.", "urgency": "Low"})
        urgency_label, urgency_msg = urgency_map[info["urgency"]]
        uc  = urgency_colors[info["urgency"]]
        ubg = urgency_bg[info["urgency"]]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Assessment</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="card-result card-disease"><div class="metric-label">Predicted Condition</div><div class="metric-value" style="color:#00e676;">{disease}</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="card-result card-confidence"><div class="metric-label">Confidence Score</div><div class="metric-value" style="color:#40c4ff;">{confidence}%</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="card-result card-urgency-{info["urgency"].lower()}"><div class="metric-label">Urgency Level</div><div class="metric-value" style="color:{uc};">{urgency_label}</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="urgency-banner" style="background:{ubg}; border-left-color:{uc};"><strong style="color:{uc};">When to see a doctor:</strong> <span style="color:#aaaaaa;">{urgency_msg}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Health Guidance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="tip-box"><div class="tip-title">Diet Recommendation</div><div class="tip-text">{info["diet"]}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="tip-box"><div class="tip-title">Lifestyle Tips</div><div class="tip-text">{info["lifestyle"]}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Differential Diagnosis</div>', unsafe_allow_html=True)

        top3_idx = np.argsort(pred_proba)[-3:][::-1]
        for i in top3_idx:
            name = le.inverse_transform([i])[0]
            prob = round(pred_proba[i] * 100, 2)
            st.markdown(f'<div class="progress-row"><div class="progress-name">{name}</div><div class="progress-track"><div class="progress-fill" style="width:{prob}%;"></div></div><div class="progress-pct">{prob}%</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">This tool is not a substitute for professional medical advice. Please consult a qualified doctor.</div>', unsafe_allow_html=True)