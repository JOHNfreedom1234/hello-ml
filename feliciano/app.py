import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
from google.generativeai import GenerativeModel

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load labels
with open('skin_cancer_model/labels.json', 'r') as f:
    labels = json.load(f)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="skin_cancer_model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

from dotenv import load_dotenv
load_dotenv()  # Loads .env variables

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(model_name="gemini-2.0-flash")

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    return input_data

def classify_image(image_path):
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    label_id = int(np.argmax(output_data))
    return label_id

def get_disease_info(disease_name):
    prompt = f"What is {disease_name}? Give a short, simple explanation for general users."
    response = gemini.generate_content(prompt)
    return response.text.strip()

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return redirect(request.url)

        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        label_id = classify_image(filepath)
        disease_name = labels[label_id]["name"] if 0 <= label_id < len(labels) else "Unknown"
        description = get_disease_info(disease_name)

        return render_template('result.html', image_url=filepath, disease_name=disease_name, description=description)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)