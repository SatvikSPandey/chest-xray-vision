import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import load_model, predict
from preprocessing import preprocess_image
from gradcam import generate_gradcam

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChestVision AI",
    page_icon="🫁",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0;
}

.main-subtitle {
    font-size: 1.1rem;
    color: #666;
    margin-top: 0.5rem;
}

.result-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    border: 2px solid #dee2e6;
}

.result-normal {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-color: #28a745;
}

.result-pneumonia {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-color: #dc3545;
}

.result-label {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.result-normal .result-label { color: #155724; }
.result-pneumonia .result-label { color: #721c24; }

.result-confidence {
    font-size: 1.1rem;
    color: #495057;
    margin-top: 0.5rem;
}

.image-label {
    text-align: center;
    font-weight: 600;
    font-size: 0.9rem;
    color: #495057;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.disclaimer {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #856404;
    margin-top: 1rem;
}

.info-box {
    background: #e7f3ff;
    border: 1px solid #b8daff;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #004085;
    margin-bottom: 1rem;
}

.stProgress > div > div > div > div {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">🫁 ChestVision AI</p>
    <p class="main-subtitle">
        Deep learning-powered chest X-ray analysis using EfficientNet-B0 with Grad-CAM explainability
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load Model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    **ChestVision AI** classifies chest X-rays as:
    - ✅ **Normal** — healthy lung tissue
    - ⚠️ **Pneumonia** — signs of infection detected
    
    ---
    
    **How it works:**
    1. Upload a chest X-ray image
    2. EfficientNet-B0 CNN analyses the image
    3. Grad-CAM highlights the regions that influenced the diagnosis
    
    ---
    
    **Model Details:**
    - Architecture: EfficientNet-B0
    - Training data: 5,216 chest X-rays
    - Classes: Normal / Pneumonia
    - Explainability: Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    ---
    
    **Built by:** [Satvik Pandey](https://github.com/SatvikSPandey)
    """)

    st.markdown("---")
    st.markdown("### 🧪 Sample Images")
    st.markdown("Don't have an X-ray? Download samples from the dataset:")
    st.markdown("[Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)")

# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("### Upload a Chest X-Ray")

st.markdown("""
<div class="info-box">
    Upload a frontal chest X-ray image (JPEG or PNG). The model will classify it as 
    Normal or Pneumonia and highlight the regions that influenced the prediction using Grad-CAM.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a frontal chest X-ray in JPEG or PNG format"
)

if uploaded_file is not None:

    # ── Preprocessing ──────────────────────────────────────────────────────────
    with st.spinner("Preprocessing image..."):
        original_img, preprocessed_img, preprocessed_display = preprocess_image(uploaded_file)

    # ── Inference ──────────────────────────────────────────────────────────────
    with st.spinner("Running AI analysis..."):
        class_name, class_index, confidence, probabilities = predict(model, preprocessed_img)

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    with st.spinner("Generating Grad-CAM explanation..."):
        heatmap_overlay, heatmap_only = generate_gradcam(model, preprocessed_img, class_index)

    # ── Result Card ────────────────────────────────────────────────────────────
    result_class = "result-normal" if class_name == "NORMAL" else "result-pneumonia"
    result_icon  = "✅" if class_name == "NORMAL" else "⚠️"

    st.markdown(f"""
    <div class="result-card {result_class}">
        <p class="result-label">{result_icon} {class_name}</p>
        <p class="result-confidence">Confidence: {confidence * 100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence Bars ────────────────────────────────────────────────────────
    st.markdown("#### Prediction Probabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Normal**")
        st.progress(float(probabilities[0]))
        st.markdown(f"`{probabilities[0]*100:.1f}%`")
    with col2:
        st.markdown("**Pneumonia**")
        st.progress(float(probabilities[1]))
        st.markdown(f"`{probabilities[1]*100:.1f}%`")

    st.divider()

    # ── Image Display ──────────────────────────────────────────────────────────
    st.markdown("#### Visual Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="image-label">Original X-Ray</p>', unsafe_allow_html=True)
        st.image(original_img, use_container_width=True)

    with col2:
        st.markdown('<p class="image-label">Preprocessed (Model Input)</p>', unsafe_allow_html=True)
        st.image(preprocessed_display, use_container_width=True)

    with col3:
        st.markdown('<p class="image-label">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
        st.image(heatmap_overlay, use_container_width=True)

    # ── Grad-CAM Explanation ───────────────────────────────────────────────────
    st.markdown("""
    **Reading the Grad-CAM heatmap:**
    🔴 **Red/Yellow regions** — high activation (most influential for the prediction)  
    🔵 **Blue regions** — low activation (less influential)
    """)

    st.divider()

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
        It is not a substitute for professional medical diagnosis. Always consult a qualified 
        radiologist or physician for medical decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty State ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #aaa;">
        <p style="font-size: 4rem;">🫁</p>
        <p style="font-size: 1.2rem;">Upload a chest X-ray to begin analysis</p>
        <p style="font-size: 0.9rem;">Supported formats: JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)