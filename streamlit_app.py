import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the trained model
model_A = load_model('mymodel.h5')

# Define the emotions dictionary
emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}

def predict_emotion(audio_data, sample_rate):
    mfcc_features = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    test_point = np.reshape(mfcc_features, newshape=(1, 40, 1))
    predictions = model_A.predict(test_point)
    result = emotions[np.argmax(predictions[0]) + 1]
    return result

def display_waveform_and_spectrogram(audio_data, sample_rate):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    
    # Plot waveform
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=axes[0])
    axes[0].set_title('Waveform')
    
    # Plot spectrogram
    D = librosa.stft(audio_data)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title('Spectrogram')
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Speech Emotion Recognition")
    
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Load audio file
        audio_data, sample_rate = librosa.load(uploaded_file)
        
        # Display waveform and spectrogram
        display_waveform_and_spectrogram(audio_data, sample_rate)
        
        # Predict emotion
        emotion = predict_emotion(audio_data, sample_rate)
        st.success(f"Predicted Emotion: {emotion}")
        
        # Display image corresponding to the emotion
        image_path = f'{emotion.lower()}.jpg'
        try:
            img = plt.imread(image_path)
            st.image(img, caption=emotion, use_column_width=True)
        except FileNotFoundError:
            st.error("Image not found for this emotion.")

if __name__ == "__main__":
    main()
