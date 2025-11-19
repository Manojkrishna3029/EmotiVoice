import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf, customtkinter as ctk, matplotlib.pyplot as plt, sounddevice as sd, wavio, time, pygame, spotipy, webbrowser, os, requests, librosa, numpy as np, threading
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
from tkinter import font
import subprocess


# Global variables
current_audio_file = None
recorded_audio_file = 'recording.wav'
image_label = None
emotion_text_label = None
invoice_text = None
waveform_canvas = None
invoice_text_widget = None  # Define invoice_text_widget as a global variable
language_entry = None
playlists = []
show_waveform_button = None
show_invoice_button = None
menu_fg_color = "#000000"  # Black color for menubar and menu items
men_fg_color = "#ffffff"
content_frame=None
predicted_emotion=None
upload_frame=None

    
# Function to show information message
def show_info(message):
    messagebox.showinfo("Information", message)

# Function to show error message
def show_error(message):
    messagebox.showerror("Error", message)


def open_main_gui():
    global root, middle_frame, main_frame, upload_frame
    middle_frame.pack_forget()  # Hide the middle frame
    main_frame.pack(fill="both", expand=True)  # Show the main frame
    root.title("Emotion Recognition")
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    # Load images once
    menu_items_top = [
        ("Home", "C:/Users/annap/project/static/home.jpeg"),
        ("Abstract", "C:/Users/annap/project/static/abstract.jpeg"),
        ("About Us", "C:/Users/annap/project/static/About.jpeg"),
        ("Project", "C:/Users/annap/project/static/project.jpg"),
        ("Help", "C:/Users/annap/project/static/settings.jpeg"),    
    ]
    menu_items_bottom = [
        ("Conclusion", "C:/Users/annap/project/static/conclusion.jpeg"),
        ("Exit", "C:/Users/annap/project/static/exit.jpg"),
    ]
    frame = ctk.CTkFrame(main_frame, fg_color=menu_fg_color)
    frame.pack(fill="y", side="left")
    profile_image = ctk.CTkLabel(frame, text="🔵", font=("Arial", 40), fg_color=menu_fg_color, text_color="white")
    profile_image.pack(pady=20)
    app_name = ctk.CTkLabel(frame, text="EmotiVoice", font=("Arial", 20), fg_color=menu_fg_color, text_color="white")
    app_name.pack(pady=10)
    labels = []
    active_text = ctk.StringVar()
    content_frame = ctk.CTkFrame(main_frame, fg_color=men_fg_color)
    content_frame.pack(fill="both", expand=True, padx=20, pady=20)
    def on_menu_click(item):
        for label in labels:
            label[1].configure(fg_color=menu_fg_color, text_color="white")
        item[1].configure(fg_color="grey", text_color="black", width=70)
        active_text.set(item[1].cget("text"))
        page_name = item[1].cget("text")
        for widget in content_frame.winfo_children():
            widget.destroy()
        if page_name == "Home":
            label = ctk.CTkLabel(content_frame, text="Home", justify="left", font=("Times New Roman", 28, "bold"))
            label.place(x=10, y=40)
            custom_font = ctk.CTkFont(family="Times New Roman", size=18)
            label = ctk.CTkLabel(content_frame, text="", font=custom_font, justify="left", anchor="nw")
            label.pack(pady=80, padx=40, fill='both', expand=True)
            tk_font = font.Font(family="Times New Roman", size=20)
            def auto_type(text, idx=0, current_text=""):
                try:
                    if idx < len(text):
                        new_char = text[idx]
                        current_line = current_text.split('\n')[-1]
                        current_line_width = tk_font.measure(current_line + new_char)
                        if current_line_width > label.winfo_width():
                            current_text += '\n'
                        current_text += new_char
                        label.configure(text=current_text)
                        idx += 1
                        content_frame.after(100, auto_type, text, idx, current_text)
                except tk.TclError:
                    # Handle TclError if label or other widgets are destroyed prematurely
                    pass
            auto_type("""Welcome to EmotiVoice!
                      This innovative project harnesses the power of emotion recognition from voice to enhance your daily experiences. By analyzing the emotional nuances in your speech, EmotiVoice can accurately detect your mood and emotional state. Whether you're feeling joyful, melancholic, energetic, or relaxed, EmotiVoice uses this information to recommend personalized playlists that perfectly match your current mood. Moreover, it suggests engaging games tailored to lift your spirits or help you unwind, ensuring an immersive and emotionally responsive user experience. Dive into a world where your emotions guide your entertainment choices, making every interaction with EmotiVoice uniquely attuned to your feelings.""")
        elif page_name == "Abstract":
            ctk.CTkLabel(content_frame, text="Abstract", font=("Times New Roman", 28, "bold")).pack(pady=40)
            abstract_text = """
            Emotion plays a crucial role in human communication, significantly impacting our interactions and decision-making processes. This project explores the innovative integration of Speech Emotion Recognition (SER) technology with personalized entertainment recommendations. By analyzing vocal inputs, the system identifies the speaker's emotional state using advanced machine learning algorithms. Once the emotion is detected, the system recommends tailored music playlists and games that align with the user's current mood. This fusion of SER with personalized entertainment aims to enhance user experiences by providing emotionally congruent content, fostering emotional well-being, and potentially offering therapeutic benefits. Our approach leverages a robust dataset for training the SER model and employs collaborative filtering techniques for recommendation, ensuring relevance and personalization. This project not only showcases the potential of emotion-aware technologies but also paves the way for more empathetic and responsive digital environments.
            """
            text_widget = ctk.CTkLabel(content_frame, text=abstract_text.strip(), wraplength=800, justify="left", font=("Times New Roman", 20))
            text_widget.pack(pady=20, padx=10)
            image_path = "C:/Users/annap/project/static/ab.jpg"
            original_image = Image.open(image_path)
            resized_image = original_image.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            label = ctk.CTkLabel(content_frame, image=photo, text="")
            label.image = photo
            label.pack()
        elif page_name == "Project":
            subprocess.run(["python", "p.py"])
        elif page_name == "Conclusion":
            ctk.CTkLabel(content_frame, text="Conclusion", font=("Times New Roman", 28, "bold")).pack(pady=40)
            conclusion_text = """
            The integration of Speech Emotion Recognition (SER) technology with personalized playlist and game recommendations marks a significant advancement in creating emotionally intelligent digital environments. By accurately analyzing vocal inputs to detect user emotions, our system offers tailored content that aligns with the user's current mood, thereby enhancing the overall user experience and contributing to emotional well-being. The combination of robust machine learning algorithms for emotion detection and collaborative filtering techniques for recommendation ensures that the content provided is both relevant and personalized, addressing the unique emotional states and preferences of each user.This project demonstrates the transformative potential of emotion-aware technologies in various applications, from entertainment to mental health support. By offering emotionally congruent content, such systems can enhance user engagement, aid in stress relief, and support mood regulation. Future developments could expand the range of emotions recognized and further refine recommendation algorithms, deepening personalization and relevance. Our work underscores the importance of integrating empathy into technology design, paving the way for more responsive and human-centered digital experiences.
            """
            text_widget = ctk.CTkLabel(content_frame, text=conclusion_text.strip(), wraplength=800, justify="left", font=("Times New Roman", 20))
            text_widget.pack(pady=20, padx=10)
            image_path = "C:/Users/annap/project/static/refresh.jpg"
            original_image = Image.open(image_path)
            resized_image = original_image.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            label = ctk.CTkLabel(content_frame, image=photo, text="")
            label.image = photo
            label.pack()
        elif page_name == "About Us":
            ctk.CTkLabel(content_frame, text="About Us", font=("Times New Roman", 28, "bold")).pack(pady=40)
            about_us_text = """
            At EmotiVoice, we are passionate about creating technology that understands and responds to human emotions. Our multidisciplinary team brings together expertise in machine learning, speech processing, and user experience design to develop innovative solutions that enhance everyday interactions. We believe in the power of empathy-driven technology to improve well-being and foster deeper connections between people and their digital environments. Our mission is to push the boundaries of what's possible in emotion recognition and personalized recommendations, making technology more intuitive, responsive, and attuned to the emotional nuances of human communication.
            """
            text_widget = ctk.CTkLabel(content_frame, text=about_us_text.strip(), wraplength=800, justify="left", font=("Times New Roman", 20))
            text_widget.pack(pady=20, padx=10)
            image_path = "C:/Users/annap/project/static/refresh.jpg"
            original_image = Image.open(image_path)
            resized_image = original_image.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            label = ctk.CTkLabel(content_frame, image=photo, text="")
            label.image = photo
            label.pack()
        elif page_name == "Help":  
            ctk.CTkLabel(content_frame, text="Help", font=("Arial", 20)).pack(pady=20)
            # Define colors and styles
            bg_color = "#000080"
            card_color = "#FFFFFF"
            form_color = "#99D0F0"
            form_frame = ctk.CTkFrame(content_frame, fg_color=bg_color, width=600, height=300)
            form_frame.place(x=500, y=200)
            ctk.CTkLabel(form_frame, text="Contact Us", fg_color=bg_color, text_color="white", font=("Arial", 28)).pack(pady=30)
            name_entry = ctk.CTkEntry(form_frame, placeholder_text="Enter your Name", width=300)
            name_entry.pack(padx=20,pady=5)
            email_entry = ctk.CTkEntry(form_frame, placeholder_text="Enter a valid email address", width=300)
            email_entry.pack(padx=70,pady=5)
            message_entry = ctk.CTkTextbox(form_frame, height=150, width=300)
            message_entry.pack(padx=50,pady=15)
            def clear_form():
                name_entry.delete(0, ctk.END)
                email_entry.delete(0, ctk.END)
                message_entry.delete("1.0", ctk.END)
                submitted_label.pack(pady=5)  # Show "Submitted" message
            submit_button = ctk.CTkButton(form_frame, text="SUBMIT", command=clear_form)
            submit_button.pack(pady=10)
            submitted_label = ctk.CTkLabel(form_frame, text="Submitted", fg_color=bg_color, font=("Arial", 16), text_color="white")
            frame_office = ctk.CTkFrame(content_frame, fg_color=form_color, width=150, height=120)
            frame_phone = ctk.CTkFrame(content_frame, fg_color=form_color, width=150, height=120)
            frame_fax = ctk.CTkFrame(content_frame, fg_color=form_color, width=150, height=120)
            frame_email = ctk.CTkFrame(content_frame, fg_color=form_color, width=150, height=120)
            frame_office.place(x=380, y=100)
            frame_phone.place(x=550, y=100)
            frame_fax.place(x=730, y=100)
            frame_email.place(x=910, y=100)
            def load_image(file_path, width, height):
                image = Image.open(file_path)
                image = image.resize((width, height), Image.LANCZOS)
                return ImageTk.PhotoImage(image)
            office_image_path = "C:/Users/annap/OneDrive/Desktop/location.jpeg"
            phone_image_path = "C:/Users/annap/OneDrive/Desktop/contact.jpeg"
            fax_image_path = "C:/Users/annap/OneDrive/Desktop/fax.jpeg"
            email_image_path = "C:/Users/annap/OneDrive/Desktop/mail.png"
            office_image = load_image(office_image_path, 50, 50)
            phone_image = load_image(phone_image_path, 50,50)
            fax_image = load_image(fax_image_path, 50, 50)
            email_image = load_image(email_image_path, 50, 50)
            ctk.CTkLabel(frame_office, image=office_image,text="", justify="center", fg_color=form_color).place(x=50, y=5)
            ctk.CTkLabel(frame_phone, image=phone_image,text="", justify="center", fg_color=form_color).place(x=60, y=5)
            ctk.CTkLabel(frame_fax, image=fax_image,text="", justify="center", fg_color=form_color).place(x=55, y=20)
            ctk.CTkLabel(frame_email, image=email_image,text="", justify="center", fg_color=form_color).place(x=58, y=20)
            ctk.CTkLabel(frame_office, text="OUR MAIN OFFICE", justify="center", fg_color=form_color, font=("Arial", 14, "bold")).place(x=10,y=50)
            ctk.CTkLabel(frame_office, text="SoHo 94 Broadway St\nNew York, NY 1001", justify="center", fg_color=form_color).place(x=10,y=70)
            ctk.CTkLabel(frame_phone, text="PHONE NUMBER", justify="center", fg_color=form_color, font=("Arial", 14, "bold")).place(x=20,y=45)
            ctk.CTkLabel(frame_phone, text="234-9876-5400\n888-0123-4567 \n(Toll Free)", justify="center", fg_color=form_color).place(x=30,y=70)
            ctk.CTkLabel(frame_fax, text="FAX", justify="center", fg_color=form_color, font=("Arial", 14, "bold")).place(x=60,y=60)
            ctk.CTkLabel(frame_fax, text="1-234-567-8900", justify="center", fg_color=form_color).place(x=30,y=80)
            ctk.CTkLabel(frame_email, text="EMAIL", justify="center", fg_color=form_color, font=("Arial", 14, "bold")).place(x=60,y=60)
            ctk.CTkLabel(frame_email, text="hello@theme.com", justify="center", fg_color=form_color).place(x=30,y=80)
        elif page_name == "Exit":
            root.destroy()
    def on_label_enter(event, label):
        if label.cget("text") != active_text.get():
            label.configure(fg_color=menu_fg_color, text_color="yellow")
    def on_label_leave(event, label):
        if label.cget("text") != active_text.get():
            label.configure(fg_color=menu_fg_color, text_color="white")
    def create_menu_item(menu_item):
        label = ctk.CTkLabel(frame, text=menu_item[0], fg_color=menu_fg_color, text_color="white", font=("Arial", 20), width=140, height=60, image=ctk.CTkImage(Image.open(menu_item[1]), size=(20, 20)), compound="left", padx=10, anchor="w", corner_radius=0)
        label.pack(fill="both", pady=5)
        label.bind("<Button-1>", lambda e: on_menu_click((menu_item, label)))
        label.bind("<Enter>", lambda e: on_label_enter(e, label))
        label.bind("<Leave>", lambda e: on_label_leave(e, label))
        return (menu_item, label)
    for item in menu_items_top + menu_items_bottom:
        labels.append(create_menu_item(item))
    on_menu_click(labels[0])
def switch_to_middle_frame():
    main_frame.pack_forget()  # Hide the main frame
    middle_frame.pack(fill="both", expand=True)  # Show the middle frame

# Create main root window
root = ctk.CTk()
root.title("Emotion Recognition")
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
middle_frame = ctk.CTkFrame(root)
main_frame = ctk.CTkFrame(root)
middle_frame.pack(fill="both", expand=True)
def close():
    root.destroy()
image_path = "C:/Users/annap/project/static/bg.png"
img = Image.open(image_path)
photo = ctk.CTkImage(light_image=img, dark_image=img, size=(root.winfo_screenwidth(), root.winfo_screenheight()))
label = ctk.CTkLabel(middle_frame, image=photo, text="")
label.image = photo
label.place(x=0, y=0, relwidth=1, relheight=1)
middle_button = ctk.CTkButton(middle_frame, text="Proceed", command=open_main_gui)
middle_button.place(x=620,y=600)
exit_button = ctk.CTkButton(middle_frame, text="exit", command=close)
exit_button.place(x=770,y=600)
middle_frame.pack(fill="both", expand=True)
root.mainloop()
