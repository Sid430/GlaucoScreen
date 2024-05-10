from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_session import Session  # You may need to install Flask-Session
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib  # For loading the logistic regression model and scaler

app = Flask(__name__)

app.secret_key = 'your_secret_key'  # You should choose a secure secret key
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)

# Define paths
UPLOAD_FOLDER = 'uploads'
DETECTION_MODEL_PATH = 'glaucoma_detection.h5'
PROGRESSION_MODEL_PATH = 'progression_detection.h5'
LOG_REG_MODEL_PATH = 'log_reg_model.pkl'
SCALER_PATH = 'scaler.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
detection_model = load_model(DETECTION_MODEL_PATH)
progression_model = load_model(PROGRESSION_MODEL_PATH)
log_reg_model = joblib.load(LOG_REG_MODEL_PATH)  # Logistic Regression Model
scaler = joblib.load(SCALER_PATH)  # Scaler for Logistic Regression Model inputs

def is_logged_in():
    return 'logged_in' in session and session['logged_in']

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if not is_logged_in():
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        age = request.form.get('age', type=float)
        gender = request.form.get('gender', type=int)  # Assuming 1 for male, 0 for female
        
        if file and allowed_file(file.filename) and age is not None and gender is not None:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform image analysis detection
            detection_result, image_confidence = predict_detection(filepath)
            
            # Perform logistic regression for tabular data
            _, log_reg_confidence = predict_log_reg(age, gender)
            
            # Combine results with weighted confidence
            combined_confidence = (3 * image_confidence + log_reg_confidence) / 4
            
            if combined_confidence >= 50:
                progression_result, progression_confidence = predict_progression(filepath)
                result = f'Glaucoma detected with a combined confidence of {combined_confidence:.2f}%. ' \
                         f'Progression status: {progression_result} with {progression_confidence:.2f}% confidence.'
            else:
                result = f'No glaucoma detected with a combined confidence of {100 - combined_confidence:.2f}%.'
                
            return render_template('result.html', prediction=result)
    return render_template('upload.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Here you should validate the username and password (hard-coded for demonstration)
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('upload_and_predict'))
        else:
            flash('Invalid login credentials!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

def predict_detection(image_path):
    img = process_image(image_path)
    prediction = detection_model.predict(img)
    confidence = np.max(prediction) * 100  # Convert to percentage
    if np.argmax(prediction) == 0:
        confidence = 100 - confidence
    return ('glaucoma' if np.argmax(prediction) == 1 else 'normal', confidence)

def predict_progression(image_path):
    img = process_image(image_path)
    prediction = progression_model.predict(img)
    confidence = np.max(prediction) * 100  # Convert to percentage
    return ('stable' if np.argmax(prediction) == 1 else 'worsening', confidence)

def predict_log_reg(age, gender):
    user_input = np.array([[age, gender]])
    user_input_scaled = scaler.transform(user_input)
    prediction = log_reg_model.predict(user_input_scaled)
    confidence = log_reg_model.predict_proba(user_input_scaled)[0][int(prediction[0])] * 100
    if prediction[0] == 0:
        confidence = 100 - confidence
    return ('glaucoma' if prediction == 1 else 'normal', confidence)

def process_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
