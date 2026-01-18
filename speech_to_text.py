from flask import Flask, render_template, request, redirect, url_for, session
import speech_recognition as sr
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to ANY random string

# Example ground truth text (for demo only)
ground_truth_texts = {
    "en": ["hello", "good morning", "how are you"],
    "hi": ["नमस्ते", "सुप्रभात", "आप कैसे हैं"],
    "te": ["హలో", "శుభోదయం", "మీరు ఎలా ఉన్నారు"],
    "ta": ["வணக்கம்", "காலை வணக்கம்", "நீங்கள் எப்படி இருக்கிறீர்கள்"],
    "kn": ["ಹಲೋ", "ಶುಭೋದಯ", "ನೀವು ಹೇಗಿದ್ದೀರಿ"]
}

# Mock CNN model for demonstration
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=100),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

text_model = create_model()
tokenizer = Tokenizer()

# Dummy login database
users = {'admin': 'admin'}

# SPEECH FUNCTION
def recognize_speech(language='en-IN'):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Say Something...')
        audio = r.listen(source)
        print('Done listening.')

    try:
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return 'Could not understand audio.'
    except sr.RequestError:
        return 'Google API unavailable.'
    except Exception as e:
        return f'Error: {e}'


# ---------------- ROUTES ---------------- #

@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        return "Login failed. Try again."

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username not in users:
            users[username] = password
            return redirect(url_for('login'))
        return "Registration failed. User exists."

    return render_template('register.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        lang = request.form['language']
        session['selected_language'] = lang
        return redirect(url_for('speech'))

    return render_template('dashboard.html')


@app.route('/speech', methods=['GET'])
def speech():
    if 'username' not in session:
        return redirect(url_for('login'))

    selected_language = session.get('selected_language', 'en')
    recognized_text = recognize_speech(language=f'{selected_language}-IN')

    sequences = tokenizer.texts_to_sequences([recognized_text])
    padded = pad_sequences(sequences, maxlen=100)
    result = text_model.predict(padded)

    # MSE simulation:
    actual_list = ground_truth_texts.get(selected_language, [""])
    if not actual_list:
        mse = "N/A"
    else:
        true_text = np.random.choice(actual_list)
        mse = np.mean([(len(true_text) - len(recognized_text)) ** 2])

    return render_template(
        'speech.html',
        text=recognized_text,
        language=selected_language,
        result=result[0][0],
        mse=mse
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('welcome'))


# RUN SERVER
if __name__ == '__main__':
    app.run(port=5003)
