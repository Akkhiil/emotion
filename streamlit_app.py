import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import urllib.request

#Define the emotions dictionary
emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}

# Function to display waveform and image
def display_waveform_and_image(file_path, emotion):
    y, sr = librosa.load(file_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot waveform
    axes[0].plot(y)
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')

    # Plot spectrogram
    axes[1].specgram(y, NFFT=1024, Fs=sr, noverlap=512)
    axes[1].set_title('Spectrogram')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency')

    # Load and display image corresponding to the emotion
    image_path = f'/content/drive/MyDrive/{emotion.lower()}.jpg'
    try:
        img = plt.imread(image_path)
        axes[2].imshow(img)
        axes[2].axis('off')
        axes[2].set_title('Emotion: ' + emotion)
    except FileNotFoundError:
        st.error("Image not found for this emotion.")

    st.pyplot(fig)

# Function to extract MFCC features
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path)
    mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc_features

# Function to predict emotion
def predict(model, file_path):
    mfcc_features = extract_mfcc(file_path)
    test_point = np.reshape(mfcc_features, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('/content/drive/MyDrive/mymodel.h5')

def application():
    model = load_model()
    models_load_state = st.text("Models Loaded")

    file_to_be_uploaded = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')

        # Save the uploaded file temporarily
        with open(file_to_be_uploaded.name, 'wb') as f:
            f.write(file_to_be_uploaded.getbuffer())

        emotion = predict(model, file_to_be_uploaded.name)
        st.success('Emotion of the audio is: ' + emotion)
        display_waveform_and_image(file_to_be_uploaded.name, emotion)

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Akkhiil/Emotion-recognition-main/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option...',
        ('Emotion Recognition', 'View Source Code')
    )

    if selected_box == 'Emotion Recognition':
        st.sidebar.success('To try by yourself by adding an audio file.')
        application()
    elif selected_box == 'View Source Code':
        st.code(get_file_content_as_string("app.py"))

if __name__ == "__main__":
    main()
