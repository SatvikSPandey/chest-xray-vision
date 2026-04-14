import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam(model, preprocessed_img, class_index):
    """
    Generates a Grad-CAM heatmap overlay on the input image.

    Args:
        model           : trained Keras model
        preprocessed_img: numpy array of shape (1, 224, 224, 3)
        class_index     : 0 for NORMAL, 1 for PNEUMONIA

    Returns:
        heatmap_overlay : uint8 RGB image with heatmap superimposed
        heatmap_colored : raw coloured heatmap
    """

    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model")

    # Build sub-model: inputs → last conv layer + final output
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Record gradients during forward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_img)
        # For binary sigmoid: use the single output neuron directly
        loss = predictions[:, 0]

    # Compute gradients of output with respect to conv layer
    grads = tape.gradient(loss, conv_outputs)

    # Global average pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by their gradients
    conv_outputs  = conv_outputs[0]
    heatmap       = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap       = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to 224x224
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Apply JET colormap — blue=low, red=high activation
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Reconstruct display version of input for overlay
    # Reverse EfficientNet preprocessing approximately for display
    input_display = np.clip(preprocessed_img[0] + 127.5, 0, 255).astype(np.uint8)

    # Superimpose heatmap with 40% opacity
    heatmap_overlay = cv2.addWeighted(input_display, 0.6, heatmap_colored, 0.4, 0)

    return heatmap_overlay, heatmap_colored