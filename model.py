import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import streamlit as st

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
THRESHOLD   = 0.45

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'weights',
    'chestvision_model.h5'
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def predict(model, preprocessed_img):
    prob_pneumonia = float(model.predict(preprocessed_img, verbose=0)[0][0])
    prob_normal    = 1.0 - prob_pneumonia

    class_index   = 1 if prob_pneumonia >= THRESHOLD else 0
    class_name    = CLASS_NAMES[class_index]
    confidence    = prob_pneumonia if class_index == 1 else prob_normal
    probabilities = [prob_normal, prob_pneumonia]

    return class_name, class_index, confidence, probabilities