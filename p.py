import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import customtkinter as ctk
import matplotlib.pyplot as plt
import sounddevice as sd
import wavio
import time
import pygame
import spotipy
import webbrowser
import os
import requests
import librosa
import numpy as np
import threading
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime

pygame.mixer.init()

model = tf.keras.models.load_model('emotion_recognition_model.h5')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMAGE_DIR = 'C:/Users/annap/project/static/'

# Spotify API credentials
client_id = 'aaaa058c985d4a0a8c4748f3e2a141ad'
client_secret = '011e082ed763413ab497194745df9a10'
redirect_uri = 'http://localhost:8888/callback'

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Global variables
current_audio_file = None
recorded_audio_file = 'recording.wav'
image_label = None
emotion_text_label = None
invoice_text = None
waveform_canvas = None
invoice_text_widget = None
language_entry = None
playlists = []
show_waveform_button = None
show_invoice_button = None
menu_fg_color = "#000000"
men_fg_color = "#ffffff"
content_frame = None
predicted_emotion = None
upload_frame = None
emotion_game_map = {
    "happy": [("Snake", "http://slither.com/io"),
              ("fighting", "https://krunker.io/")],
    "sad": [("Guess the word", "https://skribbl.io/"),
            ("Tetris Effect", "https://store.steampowered.com/app/1003590/Tetris_Effect_Connected/")],
    "angry": [("shellshock", "https://shellshock.io/"),
              ("DOOM", "https://store.steampowered.com/app/379720/DOOM/")],
    "disgust": [("Minecraft", "https://www.minecraft.net/en-us"),
               ("Overcooked! 2", "https://store.steampowered.com/app/728880/Overcooked_2/")],
    "surprise": [("Minecraft", "https://www.minecraft.net/en-us"),
                 ("Guess the word", "https://skribbl.io/")],
    "fear": [("Snake", "http://slither.com/io"),
             ("Overcooked! 2", "https://store.steampowered.com/app/728880/Overcooked_2/")],
    "neutral": [("Minecraft", "https://www.minecraft.net/en-us"),
                ("Overcooked! 2", "https://store.steampowered.com/app/728880/Overcooked_2/")]
}

def open_game_link(url):
    webbrowser.open(url)

def show_info(message):
    messagebox.showinfo("Information", message)

def show_error(message):
    messagebox.showerror("Error", message)

def display_game_recommendations(emotion):
    recommendations = emotion_game_map.get(emotion, [])
    for widget in game_recommendation_frame.winfo_children():
        widget.destroy()
    if recommendations:
        for game, url in recommendations:
            link = tk.Label(game_recommendation_frame, text=game, fg="blue", cursor="hand2", width=38, height=4)
            link.pack(pady=5)
            link.bind("<Button-1>", lambda e, url=url: open_game_link(url))
    else:
        no_recommendation = tk.Label(game_recommendation_frame, text="No game recommendations available.")
        no_recommendation.pack()

def show_waveform():
    global current_audio_file, recorded_audio_file
    file_path = current_audio_file if current_audio_file else recorded_audio_file
    feature, audio, sample_rate = extract_features(file_path)
    display_waveform_window(audio, sample_rate)

def display_waveform_window(audio, sample_rate):
    global waveform_canvas

    try:
        waveform_window = tk.Toplevel()
        waveform_window.title("Waveform Display")
        waveform_window.geometry("800x600")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(audio)
        ax1.set_title('Raw Waveform')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2)
        fig.colorbar(img, ax=ax2)
        ax2.set_title('MFCCs')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('MFCC Coefficient')

        waveform_canvas = FigureCanvasTkAgg(fig, master=waveform_window)
        waveform_canvas.draw()
        waveform_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        close_button = ctk.CTkButton(waveform_window, text="Close Waveform", command=waveform_window.destroy)
        close_button.place(x=10, y=10)

    except Exception as e:
        show_error(f"Failed to display waveform: {e}")

def close_waveform():
    global waveform_canvas

    if waveform_canvas:
        waveform_canvas.get_tk_widget().destroy()
        waveform_canvas = None
    else:
        show_info("Waveform window is not currently open.")

def show_invoice():
    global current_audio_file, recorded_audio_file
    file_path = current_audio_file if current_audio_file else recorded_audio_file
    feature, audio, sample_rate = extract_features(file_path)
    predicted_emotion = EMOTIONS[np.argmax(model.predict(np.expand_dims(feature, axis=0)))]
    generate_invoice_text(file_path, audio, sample_rate, predicted_emotion)

def generate_invoice_text(file_path, audio, sample_rate, emotion):
    global content_frame
    unique_id = str(int(time.time()))
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    amplitude = np.max(audio)
    pitch = np.mean(librosa.core.pitch_tuning(librosa.core.piptrack(y=audio, sr=sample_rate)[0]))
    duration = librosa.core.get_duration(y=audio, sr=sample_rate)
    
    global invoice_text_widget
    if invoice_text_widget:
        invoice_text_widget.destroy()

    invoice_frame = tk.Frame(content_frame, bd=10, relief=tk.GROOVE)
    invoice_frame.place(x=1300, y=200)

    header_text = f"""
    Invoice
    
    Invoice Number: {unique_id}
    Date and Time: {current_date}
    
    Description of Services:
    ---------------------------------------------------
    Item: Audio Emotion Detection
    Description: Processing and predicting emotion
    ---------------------------------------------------
    """
    
    header_label = tk.Label(invoice_frame, text=header_text, anchor="w", justify=tk.LEFT, font=("Arial", 14, "bold"))
    header_label.grid(row=0, column=0, pady=10, sticky="w")

    details = [
        ("Amplitude:", f"{amplitude:.2f}"),
        ("Pitch:", f"{pitch:.2f}"),
        ("Frequency:", f"{duration:.2f} seconds"),
        ("Predicted Emotion:", emotion.capitalize())
    ]

    for i, (label_text, value_text) in enumerate(details):
        label = tk.Label(invoice_frame, text=label_text, anchor="w", justify=tk.LEFT)
        label.grid(row=i+1, column=0, sticky="w")
        
        value_label = tk.Label(invoice_frame, text=value_text, anchor="w", justify=tk.LEFT)
        value_label.grid(row=i+1, column=1, sticky="w")

    overlay_button = ctk.CTkButton(invoice_frame, text="Close", command=close_invoice)
    overlay_button.grid(row=len(details)+2, column=0, pady=10, sticky="w")

    invoice_text_widget = invoice_frame

def close_invoice():
    global invoice_text_widget

    if invoice_text_widget:
        invoice_text_widget.destroy()
        invoice_text_widget = None
    else:
        show_info("Invoice display is not currently open.")

def display_image(emotion):
    global image_label, emotion_text_label, show_waveform_button, show_invoice_button, content_frame, playlist_listbox, language_combobox, upload_frame
    image_path = IMAGE_DIR + emotion + ".jpeg"
    image = Image.open(image_path)
    image = image.resize((300, 300), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    if image_label is None:
        image_label = tk.Label(upload_frame, image=photo)
        image_label.image = photo
        image_label.place(x=700, y=210)
    else:
        image_label.config(image=photo)
        image_label.image = photo
    if emotion_text_label is None:
        emotion_text_label = ctk.CTkLabel(upload_frame, text=f"Predicted Emotion: {emotion.capitalize()}", font=("Arial", 16))
        emotion_text_label.place(x=700, y=150)
    else:
        emotion_text_label.configure(text=f"Predicted Emotion: {emotion.capitalize()}")
    if show_waveform_button is None:
        show_waveform_button = ctk.CTkButton(upload_frame, text="Show Waveform", command=show_waveform)
        show_waveform_button.place(x=700, y=550)
    if show_invoice_button is None:
        show_invoice_button = ctk.CTkButton(upload_frame, text="Generate Invoice", command=show_invoice)
        show_invoice_button.place(x=700, y=600)

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Mean across time steps
        return mfccs_mean, audio, sample_rate
    except Exception as e:
        show_error(f"Error encountered while parsing file: {file_name}, error: {e}")
        return None, None, None

def predict_emotion():
    global invoice_text, playlists, playlist_listbox, predicted_emotion
    try:
        file_path = current_audio_file if current_audio_file else recorded_audio_file
        feature, audio, sample_rate = extract_features(file_path)
        if feature is not None:
            feature = np.expand_dims(feature, axis=0)
            prediction = model.predict(feature)
            predicted_emotion = EMOTIONS[np.argmax(prediction)]
            display_image(predicted_emotion)
            refresh_playlists()  # Refresh playlists after predicting emotion
            display_game_recommendations(predicted_emotion)
            show_waveform_button.configure(state=tk.NORMAL)  # Enable show waveform button
            show_invoice_button.configure(state=tk.NORMAL)  # Enable show invoice button
            return predicted_emotion  # Return the predicted emotion
        else:
            show_error("Failed to extract features from the selected file.")
            return None
    except Exception as e:
        show_error(f"Failed to predict emotion: {e}")
        return None


def upload_audio_file():
    global current_audio_file
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        current_audio_file = file_path
        show_info(f"Audio file selected: {file_path}")
        predict_emotion()

def record_audio():
    global recorded_audio_file
    show_info("Recording started. Speak now...")
    recording_start_time = time.time()
    recording_duration = 5  # Duration in seconds
    audio_data = sd.rec(int(recording_duration * 44100), samplerate=44100, channels=2, dtype='int16')
    sd.wait()
    wavio.write(recorded_audio_file, audio_data, 44100, sampwidth=2)
    show_info("Recording stopped.")
    predict_emotion()

def get_spotify_playlists(emotion):
    try:
        selected_language = language_combobox.get()
        query = f"{emotion} {selected_language}"
        results = sp.search(q=query, type='playlist', limit=5)
        playlists = []

        for playlist in results['playlists']['items']:
            playlists.append({
                'name': playlist['name'],
                'url': playlist['external_urls']['spotify']
            })

        return playlists

    except Exception as e:
        print(f"Error retrieving playlists: {str(e)}")
        return []    
def refresh_playlists():
    global playlists, playlist_listbox, predicted_emotion
    #predicted_emotion = predict_emotion()
    if predicted_emotion:
        playlists = get_spotify_playlists(predicted_emotion)
        playlist_listbox.delete(0, tk.END)
        for playlist in playlists:
            playlist_listbox.insert(tk.END, playlist['name'])
# Function to open the playlist URL
def open_playlist_url(event):
    global playlists, current_audio_file, recorded_audio_file
    selection_index = playlist_listbox.curselection()
    if selection_index:
        url = playlists[selection_index[0]]['url']
        webbrowser.open_new(url)
def close_second_file():
    # Close the second file
    os._exit(0)
    
    # Reopen the first file
    subprocess.Popen(["python", "hii.py"])

def create_gui():
    global content_frame, upload_frame, game_recommendation_frame, playlist_listbox, language_combobox

    root = tk.Tk()
    root.title("Emotion Recognition")
    root.state("zoomed")

    content_frame = tk.Frame(root)
    content_frame.pack(fill=tk.BOTH, expand=True)

    upload_frame = tk.Frame(content_frame)
    upload_frame.pack(fill=tk.BOTH, expand=True)

    game_recommendation_frame = tk.Frame(content_frame)
    game_recommendation_frame.place(x=1300, y=150)  # Adjust positioning as needed

    upload_button = ctk.CTkButton(upload_frame, text="Upload Audio File", command=upload_audio_file)
    upload_button.place(x=10, y=10)

    record_button = ctk.CTkButton(upload_frame, text="Record Audio", command=record_audio)
    record_button.place(x=10, y=50)

    # Language Combobox
    # Language Combobox
    language_label = tk.Label(upload_frame, text="Select Language:")
    language_label.place(x=120, y=150)
   
    language_combobox = ttk.Combobox(upload_frame, values=["Telugu", "Hindi", "English", "Spanish", "French", "German", "Italian"])
    language_combobox.place(x=250, y=150)
    language_combobox.set("Telugu")

# Playlist Listbox
    playlist_listbox = tk.Listbox(upload_frame, font=("Arial", 12), width=38, height=10)
    playlist_listbox.place(x=150, y=270)
    playlist_listbox.bind("<Double-1>", open_playlist_url)

# Refresh Playlists Button
    refresh_button = tk.Button(upload_frame, text="Refresh Playlists", command=refresh_playlists)
    refresh_button.place(x=340, y=220)
    close_button = tk.Button(root, text="Back", command=close_second_file)
    close_button.pack(pady=20)


    root.mainloop()


if __name__ == "__main__":
    create_gui()
