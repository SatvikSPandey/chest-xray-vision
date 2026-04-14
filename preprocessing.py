import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224

def preprocess_image(uploaded_file):
    """
    Takes a Streamlit uploaded file object and returns:
    - original_img        : RGB numpy array for display
    - preprocessed_img    : EfficientNet-preprocessed array ready for the model
    - preprocessed_display: uint8 RGB array for showing the user what the model sees
    """

    # Open and convert to RGB
    pil_image = Image.open(uploaded_file).convert('RGB')

    # Keep original for display before resizing
    original_img = np.array(pil_image)

    # Resize using PIL — same as Keras load_img internally uses
    pil_resized = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Convert to numpy array
    resized = np.array(pil_resized).astype(np.float32)

    # Keep uint8 copy for display
    preprocessed_display = np.array(pil_resized)

    # Apply EfficientNet preprocessing
    preprocessed = preprocess_input(resized)

    # Add batch dimension
    preprocessed_img = np.expand_dims(preprocessed, axis=0)

    return original_img, preprocessed_img, preprocessed_display