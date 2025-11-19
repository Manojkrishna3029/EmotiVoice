# streamlit_app.py
import streamlit as st
import numpy as np
import tempfile
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import base64
from dotenv import load_dotenv
import os

load_dotenv() 
st.set_page_config(page_title="EmotiVoice - Online", layout="wide")

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_model(path="emotion_recognition_model.h5"):
    return tf.keras.models.load_model(path)

# Load model (will raise if file not present)
try:
    model = load_model("emotion_recognition_model.h5")
except Exception as e:
    st.error(f"Failed to load model: {e}. Make sure emotion_recognition_model.h5 is in the repo root.")
    model = None

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Spotify client using only .env variables ---
def get_spotify_client():
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        st.warning("Spotify credentials not found in .env!")
        return None

    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)
        return sp
    except Exception as e:
        st.warning(f"Failed to create Spotify client: {e}")
        return None

# Initialize Spotify client
sp = get_spotify_client()


# --- Helpers ---
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs, audio, sr
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None, None

def predict_emotion_from_features(feature):
    if model is None:
        return None
    feature = np.expand_dims(feature, axis=0)
    pred = model.predict(feature)
    emotion = EMOTIONS[int(np.argmax(pred))]
    return emotion, pred.flatten().tolist()

def plot_waveform_and_mfcc(audio, sr):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))
    axs[0].plot(audio)
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Samples")
    axs[0].set_ylabel("Amplitude")

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=axs[1])
    axs[1].set_title("MFCC")
    axs[1].set_xlabel("Time")
    fig.colorbar(img, ax=axs[1])
    plt.tight_layout()
    return fig

def get_spotify_playlists_for_emotion(emotion, limit=5):
    if sp is None:
        st.info("Spotify client not configured. Add credentials to .env.")
        return []

    mapping = {
        "happy": "party",
        "angry": "metal",
        "neutral": "ambient",
        "surprise": "pop",
        "fear": "dark",
        "disgust": "grunge",
        "sad": "mood"
    }
    q = mapping.get(emotion, emotion)

    try:
        results = sp.search(q=f"{q} playlist", type="playlist", limit=limit)
        playlists = [{"name": item['name'], "url": item['external_urls']['spotify']} 
                     for item in results.get('playlists', {}).get('items', [])]
        if not playlists:
            st.info("No playlists found for this emotion.")
        return playlists
    except Exception as e:
        st.warning(f"Spotify lookup failed: {e}")
        return []


# --- UI ---
st.title("EmotiVoice — Voice Emotion Detection (Web)")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload / Record Audio")
    uploaded = st.file_uploader("Upload a WAV file (or MP3)", type=['wav', 'mp3', 'm4a', 'flac'])
    st.markdown("Or try the sample demo audio below.")
    if st.button("Use example audio"):
        # if you want, add a small example audio file into repo and load it; here we create placeholder
        example_path = "demo_example.wav"
        if os.path.exists(example_path):
            uploaded = open(example_path, "rb")
            st.success("Loaded example audio.")
        else:
            st.info("No demo audio found in repo. Please upload your file.")

    if uploaded:
        # Save to temp file
        tmp_path = save_uploaded_file(uploaded) if hasattr(uploaded, "getbuffer") else None

        if tmp_path:
            st.audio(tmp_path)
            feature, audio, sr = extract_features(tmp_path)
            if feature is not None:
                if st.button("Predict Emotion"):
                    with st.spinner("Predicting..."):
                        emotion, probs = predict_emotion_from_features(feature)
                        if emotion:
                            st.success(f"Predicted emotion: **{emotion.upper()}**")
                            # Show confidence bar
                            probs_arr = np.array(probs)
                            df = {EMOTIONS[i]: float(probs_arr[i]) for i in range(len(EMOTIONS))}
                            st.bar_chart(df)

                            # Play waveform + MFCC
                            fig = plot_waveform_and_mfcc(audio, sr)
                            st.pyplot(fig)

                            # Spotify playlists
                            st.subheader("Recommended Playlists")
                            playlists = get_spotify_playlists_for_emotion(emotion)
                            if playlists:
                                for p in playlists:
                                    st.markdown(f"- [{p['name']}]({p['url']})")
                            else:
                                st.info("No playlists found (Spotify creds missing or failed).")

                            # Game recommendations (simple mapping)
                            emotion_game_map = {
                                "happy": ["Snake", "Krunker.io"],
                                "sad": ["Skribbl.io", "Tetris"],
                                "angry": ["Shellshock.io", "DOOM"],
                                "disgust": ["Minecraft", "Overcooked 2"],
                                "surprise": ["Minecraft", "Skribbl.io"],
                                "fear": ["Snake", "Overcooked 2"],
                                "neutral": ["Minecraft", "Overcooked 2"]
                            }
                            st.subheader("Fun game suggestions")
                            for g in emotion_game_map.get(emotion, []):
                                st.write(f"- {g}")
                        else:
                            st.error("Prediction failed.")
        else:
            st.error("Failed to save uploaded file.")

with col2:
    st.header("About / How it works")
    st.markdown("""
    **EmotiVoice** analyzes the emotional content of a short speech/audio clip using a pre-trained TensorFlow model.
    - Upload a WAV/MP3 audio file (short, clear speech works best).
    - The app extracts MFCC features, predicts emotion, shows waveform + MFCC, and recommends Spotify playlists.
    """)
    st.markdown("**Model:** `emotion_recognition_model.h5` (must be in the repo root)")
    st.markdown("**Spotify:** Optional. Add credentials to Streamlit secrets for real playlist results.")
    st.write("---")
    st.markdown("**Deployment**: Use Streamlit Cloud (share.streamlit.io). See instructions below.")
    st.write("---")
    st.markdown("**Notes / Limitations**")
    st.markdown("""
    - This web app **does not** record microphone audio from the visitor (server-side).  
    - Browser recording requires client-side JS; we use upload for compatibility.  
    - Large model files may exceed Streamlit repo limits — consider hosting model in a cloud bucket (S3) or compress model.
    """)

# Footer: show Spotify secrets status
st.write("---")
if sp is None:
    st.warning("Spotify credentials not found. To enable playlist lookup, add SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET to Streamlit secrets.")
else:
    st.success("Spotify configured.")

st.markdown("© EmotiVoice")
