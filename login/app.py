from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_mail import Mail, Message
import random
import string

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'annapurna8054@gmail.com'
app.config['MAIL_PASSWORD'] = 'Manoj@123'

mail = Mail(app)

# Temporary storage for verification codes
verification_codes = {}

def generate_verification_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Generate verification code
        verification_code = generate_verification_code()
        verification_codes[email] = verification_code

        # Send verification email
        msg = Message('Email Verification', sender='annapurna8054@gmail.com', recipients=[email])
        msg.body = f'Your verification code is {verification_code}'
        mail.send(msg)

        # Store user info in session
        session['username'] = username
        session['email'] = email
        session['password'] = password

        return redirect(url_for('verify_email'))

    return render_template('register.html')

@app.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    if request.method == 'POST':
        email = session.get('email')
        entered_code = request.form['verification_code']

        if email and verification_codes.get(email) == entered_code:
            flash('Registration successful!', 'success')
            # Clear session and verification codes
            session.clear()
            verification_codes.pop(email, None)
            return redirect(url_for('login'))
        else:
            flash('Invalid verification code', 'danger')

    return render_template('verify_email.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Add login logic here
        pass
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
