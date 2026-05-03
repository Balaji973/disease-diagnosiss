import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Disease Risk Alert",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
    <style>
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-size: 18px;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model    = joblib.load(os.path.join(BASE_DIR, 'model', 'disease_model.pkl'))
le       = joblib.load(os.path.join(BASE_DIR, 'model', 'label_encoder.pkl'))
symptoms = joblib.load(os.path.join(BASE_DIR, 'model', 'symptom_columns.pkl'))# ── Suggestions ───────────────────────────────────────────
suggestions = {
    "Diabetes": {
        "diet": "Low sugar, high fiber. Avoid white rice & sweets.",
        "lifestyle": "Walk 30 mins daily, check blood sugar weekly.",
        "urgency": "High"
    },
    "Hypertension": {
        "diet": "Low salt diet. Eat bananas and leafy greens.",
        "lifestyle": "Reduce stress, limit caffeine, sleep 8 hrs.",
        "urgency": "High"
    },
    "Anemia": {
        "diet": "Iron-rich foods: spinach, lentils, red meat.",
        "lifestyle": "Avoid tea after meals, take iron supplements.",
        "urgency": "Medium"
    },
    "Migraine": {
        "diet": "Avoid chocolate, caffeine and processed food.",
        "lifestyle": "Sleep on schedule, avoid bright screens at night.",
        "urgency": "Medium"
    },
    "Fungal infection": {
        "diet": "Avoid sugar and fermented foods.",
        "lifestyle": "Keep skin dry, wear breathable clothing.",
        "urgency": "Low"
    },
    "Allergy": {
        "diet": "Avoid known allergens, eat anti-inflammatory foods.",
        "lifestyle": "Keep environment clean, avoid dust and pollen.",
        "urgency": "Medium"
    },
    "Pneumonia": {
        "diet": "Warm fluids, vitamin C rich foods.",
        "lifestyle": "Rest completely, avoid cold air.",
        "urgency": "High"
    },
    "Dengue": {
        "diet": "Papaya leaf juice, coconut water, high fluid intake.",
        "lifestyle": "Complete bed rest, avoid mosquito exposure.",
        "urgency": "High"
    },
    "Typhoid": {
        "diet": "Boiled foods, avoid raw vegetables and street food.",
        "lifestyle": "Complete rest, stay hydrated.",
        "urgency": "High"
    },
    "Malaria": {
        "diet": "Light easily digestible foods, lots of fluids.",
        "lifestyle": "Use mosquito nets, take prescribed medication.",
        "urgency": "High"
    },
    "Acne": {
        "diet": "Avoid oily & junk food. Eat zinc-rich foods like pumpkin seeds.",
        "lifestyle": "Wash face twice daily, avoid touching face, drink 3L water.",
        "urgency": "Low"
    },
    "Impetigo": {
        "diet": "Boost immunity with Vitamin C foods like oranges and amla.",
        "lifestyle": "Keep skin clean, avoid sharing towels or clothes.",
        "urgency": "Medium"
    },
    "Jaundice": {
        "diet": "Sugarcane juice, lemon water, avoid fatty foods completely.",
        "lifestyle": "Complete bed rest, avoid alcohol, drink lots of fluids.",
        "urgency": "High"
    },
    "Chicken pox": {
        "diet": "Soft foods, coconut water, avoid spicy food.",
        "lifestyle": "Avoid scratching, stay isolated, keep skin clean.",
        "urgency": "High"
    },
}

urgency_map = {
    "High":   ("🔴 HIGH",   "See a doctor within 2 days"),
    "Medium": ("🟡 MEDIUM", "Monitor symptoms for 1 week"),
    "Low":    ("🟢 LOW",    "Maintain healthy habits")
}

urgency_colors = {
    "High": "#e74c3c",
    "Medium": "#f39c12",
    "Low": "#2ecc71"
}

# ── Header ────────────────────────────────────────────────
st.title("🏥 Early Disease Risk Alert System")
st.markdown("#### AI-powered symptom checker with urgency rating & health tips")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("ℹ️ About This App")
st.sidebar.info(
    "This tool uses a Machine Learning model "
    "trained on symptoms across 40+ diseases "
    "to predict your health risk instantly."
)
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Dataset:** Kaggle Disease-Symptom")
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ For informational purposes only. Always consult a doctor.")

# ── Main Input ────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Select Your Symptoms")
    selected = st.multiselect(
        "Choose all symptoms you are experiencing:",
        options=symptoms,
        help="Select as many symptoms as apply"
    )
    st.caption(f"✅ {len(selected)} symptom(s) selected")

with col2:
    st.subheader("👤 Basic Info")
    age    = st.slider("Your Age", 1, 100, 25)
    gender = st.radio("Gender", ["Male", "Female", "Other"])

st.markdown("---")

# ── Predict ───────────────────────────────────────────────
if st.button("🔎 Analyze My Risk"):

    if len(selected) == 0:
        st.error("⚠️ Please select at least one symptom!")

    else:
        # Build input vector
        input_vec = [1 if s in selected else 0 for s in symptoms]
        input_df  = pd.DataFrame([input_vec], columns=symptoms)

        # Predict
        pred_encoded = model.predict(input_df)[0]
        pred_proba   = model.predict_proba(input_df)[0]
        confidence   = round(max(pred_proba) * 100, 2)
        disease      = le.inverse_transform([pred_encoded])[0]

        # Get suggestions
        info = suggestions.get(disease, {
            "diet": "Eat balanced meals with fruits and vegetables.",
            "lifestyle": "Stay active, hydrated and sleep 8 hrs daily.",
            "urgency": "Low"
        })
        urgency_label, urgency_msg = urgency_map[info["urgency"]]
        uc = urgency_colors[info["urgency"]]

        # ── Result Cards ──────────────────────────────────
        st.markdown("## 📋 Your Risk Report")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
                <div style="background:#1e3a5f; padding:20px; border-radius:10px;
                border:1px solid #2ecc71; text-align:center;">
                    <p style="color:#aaaaaa; font-size:14px; margin:0;">🩺 Predicted Condition</p>
                    <p style="color:#2ecc71; font-size:26px; font-weight:bold; margin:5px 0;">{disease}</p>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
                <div style="background:#1e3a5f; padding:20px; border-radius:10px;
                border:1px solid #3498db; text-align:center;">
                    <p style="color:#aaaaaa; font-size:14px; margin:0;">📊 Confidence Score</p>
                    <p style="color:#3498db; font-size:26px; font-weight:bold; margin:5px 0;">{confidence}%</p>
                </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
                <div style="background:#1e3a5f; padding:20px; border-radius:10px;
                border:1px solid {uc}; text-align:center;">
                    <p style="color:#aaaaaa; font-size:14px; margin:0;">⚠️ Urgency Level</p>
                    <p style="color:{uc}; font-size:26px; font-weight:bold; margin:5px 0;">{urgency_label}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**When to see a doctor:** {urgency_msg}")
        st.markdown("---")

        # ── Health Tips ───────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🥗 Diet Recommendation")
            st.success(info["diet"])
        with c2:
            st.markdown("### 🏃 Lifestyle Tips")
            st.success(info["lifestyle"])

        st.markdown("---")

        # ── Top 3 Possible Conditions ─────────────────────
        st.markdown("### 📊 Top 3 Possible Conditions")
        top3_idx = np.argsort(pred_proba)[-3:][::-1]
        for i in top3_idx:
            name = le.inverse_transform([i])[0]
            prob = round(pred_proba[i] * 100, 2)
            st.progress(int(prob), text=f"{name} → {prob}%")

        st.markdown("---")
        st.caption("⚠️ Disclaimer: This is not a substitute for professional medical advice. Please consult a qualified doctor.")
