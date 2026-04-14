# 🫁 ChestVision AI

A deep learning-powered chest X-ray analysis system that classifies images as **Normal** or **Pneumonia** with Grad-CAM explainability — built to demonstrate production-grade computer vision engineering for medical imaging applications.

**Live Demo:** [chestvision-satvik.streamlit.app](https://chestvision-satvik.streamlit.app)

---

## Why This Project

Siemens Healthineers is one of the world's largest medical imaging companies. This project directly demonstrates:

- **CNN-based medical image classification** — the core of clinical AI
- **Grad-CAM explainability** — a clinical requirement; radiologists won't trust black-box AI
- **Production ML deployment** — live demo recruiters can test without installation

---

## Architecture

User uploads chest X-ray
│
▼
OpenCV Preprocessing Pipeline

PIL image loading
Resize to 224×224
EfficientNet normalization
│
▼
EfficientNet-B0 CNN (fine-tuned)
Pre-trained on ImageNet
Fine-tuned on 5,216 chest X-rays
Binary sigmoid output
│
▼
Grad-CAM Explainability Engine
Extracts gradients from last Conv2D layer
Generates activation heatmap
OpenCV JET colormap overlay
│
▼
Streamlit UI
Original X-ray
Preprocessed input
Grad-CAM heatmap
Diagnosis + confidence

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow 2.21, Keras |
| Model | EfficientNet-B0 (fine-tuned) |
| Computer Vision | OpenCV |
| Explainability | Grad-CAM |
| Training | Google Colab (T4 GPU) |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## Model Details

- **Architecture:** EfficientNet-B0 with custom classification head
- **Training data:** 5,216 chest X-rays (Kaggle — Paul Mooney)
- **Classes:** NORMAL / PNEUMONIA
- **Test accuracy:** 86.54%
- **Normal accuracy:** 67.9% | **Pneumonia accuracy:** 97.7%
- **Loss function:** Focal Loss (γ=2.0, α=0.6) — handles class imbalance
- **Training approach:** Two-phase — frozen base then fine-tuned top 20 layers

### Known Limitations

The test set contains X-rays from two different sources — adult scans (NORMAL2-IM-\*) and pediatric scans (IM-\*). The model performs better on adult scans due to distribution shift between the two sources. This is a known challenge in medical imaging AI and an active area of research.

---

## Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights which regions of the X-ray influenced the model's decision:

- 🔴 **Red/Yellow** — high activation (most influential regions)
- 🔵 **Blue** — low activation (less influential regions)

This is a clinical requirement in production medical AI systems — physicians and radiologists need to understand *why* a model made a decision before trusting it.

---

## Project Structure

chest-xray-vision/
├── app.py                          ← Streamlit UI
├── model.py                        ← Model loader and inference
├── preprocessing.py                ← OpenCV preprocessing pipeline
├── gradcam.py                      ← Grad-CAM heatmap generator
├── train.py                        ← Training script (run locally)
├── weights/
│   └── chestvision_full_model.keras ← Trained model weights
├── sample_images/
│   ├── normal_sample.jpeg
│   └── pneumonia_sample.jpeg
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md

---

## Local Setup

```bash
git clone https://github.com/SatvikSPandey/chest-xray-vision
cd chest-xray-vision
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Paul Mooney, Kaggle

---

## Medical Disclaimer

This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for medical decisions.

---

**Built by:** [Satvik Pandey](https://github.com/SatvikSPandey) | AI Engineer | Python Developer

