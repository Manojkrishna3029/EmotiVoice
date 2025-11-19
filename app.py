from flask import Flask, render_template, request, redirect, url_for, flash, session
import librosa
import numpy as np
import tensorflow as tf
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Simulate a database with a dictionary
users_db = {
    'admin': {'email': 'manojkrishna@gmail.com', 'password': generate_password_hash('Krishna@123', method='pbkdf2:sha256'), 'role': 'admin'}
}

# Load your trained emotion recognition model
model = tf.keras.models.load_model('emotion_recognition_model.h5')

# Spotify API credentials
SPOTIPY_CLIENT_ID = 'aaaa058c985d4a0a8c4748f3e2a141ad'
SPOTIPY_CLIENT_SECRET = '011e082ed763413ab497194745df9a10'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# Spotify scopes required
SPOTIFY_SCOPES = 'playlist-read-private'

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=SPOTIFY_SCOPES))

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_type = request.form['loginType']
        username = request.form['username']
        password = request.form['password']

        # Check admin credentials
        if username == 'admin' and check_password_hash(users_db['admin']['password'], password):
            session['username'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin_panel'))

        # Check user credentials
        if username in users_db and check_password_hash(users_db[username]['password'], password):
            session['username'] = username
            session['role'] = users_db[username]['role']
            if login_type == 'user' and users_db[username]['role'] == 'user':
                return redirect(url_for('menupage'))
            else:
                flash('Invalid login type for the given username')
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if username in users_db:
            flash('Username already exists')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            users_db[username] = {'email': email, 'password': hashed_password, 'role': 'user'}
            flash('Registration successful')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/admin_panel')
def admin_panel():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    return render_template('admin_panel.html')

@app.route('/menupage')
def menupage():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('menupage.html')

@app.route('/index')
def index():
    playlists = fetch_spotify_playlists(language='telugu')
    return render_template('index.html', playlists=playlists)

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'audioFile' not in request.files:
        return redirect(request.url)
    
    file = request.files['audioFile']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature = np.array([mfccs])
        prediction = model.predict(feature)
        predicted_emotion = EMOTIONS[np.argmax(prediction)]
        
        playlists = fetch_spotify_playlists(predicted_emotion)
        
        return render_template('index.html', prediction=predicted_emotion, playlists=playlists)
    
    return redirect(request.url)

def fetch_spotify_playlists(emotion=None, language=None):
    playlists = []
    try:
        if language:
            results = sp.category_playlists(category_id='telugu', limit=5)
        elif emotion:
            if emotion == 'happy':
                results = sp.category_playlists(category_id='party', limit=5)
            elif emotion == 'angry':
                results = sp.category_playlists(category_id='metal', limit=5)
            elif emotion == 'neutral':
                results = sp.category_playlists(category_id='ambient', limit=5)
            elif emotion == 'surprise':
                results = sp.category_playlists(category_id='pop', limit=5)
            elif emotion == 'fear':
                results = sp.category_playlists(category_id='dark', limit=5)
            elif emotion == 'disgust':
                results = sp.category_playlists(category_id='grunge', limit=5)
            elif emotion == 'sad':
                results = sp.category_playlists(category_id='mood', limit=5)
        
        for playlist in results['playlists']['items']:
            playlists.append({'name': playlist['name'], 'url': playlist['external_urls']['spotify']})
            
    except spotipy.SpotifyException as e:
        print(f"Error fetching playlists: {e}")

    return playlists

if __name__ == '__main__':
    app.run(debug=True)



