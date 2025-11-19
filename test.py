import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Function to extract features from an audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}, error: {e}")
        return None
    return mfccs_scaled

# Load the saved model
model_path = r"C:\Users\annap\OneDrive\Desktop\project\emotion_recognition_model.h5"
model = load_model(model_path)

# Define the emotions and encode them
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# Function to predict the emotion of a new audio file
def predict_emotion(file_name):
    feature = extract_features(file_name)
    if feature is not None:
        feature = feature.reshape(1, -1)
        prediction = model.predict(feature)
        emotion = label_encoder.inverse_transform([np.argmax(prediction)])
        return emotion[0]
    return None

# Path to the single audio file to test
single_audio_file = r"C:\Users\annap\OneDrive\Desktop\project\test dataset\YAF_angry\YAF_bar_angry.wav"  # Ensure this path is correct

# Predict emotion for the single audio file
if os.path.exists(single_audio_file):
    predicted_emotion = predict_emotion(single_audio_file)
    print(f'File: {single_audio_file}, Predicted emotion: {predicted_emotion}')
else:
    print(f"File {single_audio_file} does not exist.")
