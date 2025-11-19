import librosa
import numpy as np
import os
import glob
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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

# Define the emotions and dataset path
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
data_path = r"C:\Users\annap\OneDrive\Desktop\project\Data set"

features = []
labels = []

# Iterate through each emotion folder and extract features
for emotion in emotions:
    emotion_path = os.path.join(data_path, emotion)
    for file in glob.glob(os.path.join(emotion_path, "*.wav")):
        feature = extract_features(file)
        if feature is not None:
            features.append(feature)
            labels.append(emotion)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode the labels
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(40,), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Save the model
model_path = r"C:\Users\annap\OneDrive\Desktop\project\emotion_recognition_model.h5"
model.save(model_path)
print(f'Model saved to {model_path}')

# Function to predict the emotion of a new audio file
def predict_emotion(file_name):
    feature = extract_features(file_name)
    if feature is not None:
        feature = feature.reshape(1, -1)
        prediction = model.predict(feature)
        emotion = label_encoder.inverse_transform([np.argmax(prediction)])
        return emotion[0]
    return None

# Predict on a new file
new_file = r"C:\Users\annap\OneDrive\Desktop\project\new_audio_file.wav"  # Ensure this path is correct
if os.path.exists(new_file):
    predicted_emotion = predict_emotion(new_file)
    print(f'Predicted emotion: {predicted_emotion}')
else:
    print(f"File {new_file} does not exist.")


