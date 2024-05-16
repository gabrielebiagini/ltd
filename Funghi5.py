import streamlit as st
from PIL import Image
import numpy as np

# Definizione dei parametri
IMAGE_SIZE = 224

# Caricamento del modello addestrato
model_path = 'C:/Users/gabri/Desktop/Stream/mushroom_classifier_with_conv.h5'
model = tf.keras.models.load_model(model_path)

# Funzione per preprocessare l'immagine
def preprocess_image(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    return image

# Funzione per la classificazione del fungo
def classify_mushroom(image):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    prediction = model.predict(preprocessed_image)[0]
    predicted_class = 1 if prediction >= 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction
    return predicted_class, confidence, prediction

# Funzione per Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Funzione per visualizzare Grad-CAM
def display_gradcam(image, heatmap, alpha=0.4):
    img = np.array(image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    return Image.fromarray(np.uint8(superimposed_img))

# Interfaccia Streamlit
st.title('Classificazione Fungo con Grad-CAM')
uploaded_file = st.file_uploader('Carica una foto del fungo', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Fungo caricato', use_column_width=True)
    
    predicted_class, confidence, prediction = classify_mushroom(image)
    
    class_names = ['Mangiabile', 'Velenoso']
    st.write(f'Il fungo è: {class_names[predicted_class]}')
    st.write(f'Confidenza: {confidence[0] * 100:.2f}%')

    # Mostra confidenza per entrambe le classi
    st.write('Dettaglio Confidenze:')
    st.write(f'{class_names[0]}: {(1 - prediction[0]) * 100:.2f}%')
    st.write(f'{class_names[1]}: {prediction[0] * 100:.2f}%')

    # Genera Grad-CAM heatmap
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    heatmap = make_gradcam_heatmap(preprocessed_image, model, 'Conv_1')  # Sostituisci 'Conv_1' con il nome del layer convoluzionale corretto
    superimposed_img = display_gradcam(image, heatmap)
    
    st.write("Grad-CAM:")
    st.image(superimposed_img, caption='Grad-CAM', use_column_width=True)
