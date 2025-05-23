import os
import json
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from io import BytesIO

# Configure Flask
app = Flask(__name__)

# TensorFlow Lite configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment.")
print("✅ Gemini API Key Loaded Successfully")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Model path setup
MODEL_PATH = r"C:\Users\LENOVO\OneDrive - wvsu.edu.ph\GitHub\hello-ml\gasis\skin_cancer_model\model.tflite"
LABELS_PATH = r"C:\Users\LENOVO\OneDrive - wvsu.edu.ph\GitHub\hello-ml\gasis\skin_cancer_model\labels.txt"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load TFLite model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label names
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
with open(LABELS_PATH, 'r') as file:
    labels = [line.strip() for line in file.readlines()]
label_names = {label: label for label in labels}

# Preprocessing
def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Classification
def classify_image(image_data):
    input_data = preprocess_image(image_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_index = np.argmax(output_data)
    label = labels[pred_index]
    confidence = float(output_data[pred_index])
    return label, label_names.get(label, "Unknown"), confidence

# Gemini response handling
def get_description(label_name):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = (
        f"Provide the following information about {label_name}:\n"
        f"Description:\n"
        f"Causes (comma-separated):\n"
        f"Risk Factors (comma-separated):\n"
        f"Prognosis (just survivability percentage):\n"
        f"Treatments (one per line):"
    )

    response = model.generate_content(prompt)
    print("🧠 Gemini Raw Response:")
    print(response.text)

    # Initialize structured output
    details = {
        "description": "",
        "causes": [],
        "risk_factors": [],
        "prognosis": "",
        "treatments": []
    }

    # Parse the response
    lines = response.text.strip().splitlines()
    current = None
    for line in lines:
        if line.startswith("Description:"):
            current = "description"
            details["description"] = line.replace("Description:", "").strip()
        elif line.startswith("Causes:"):
            current = "causes"
            details["causes"] = [x.strip() for x in line.replace("Causes:", "").split(',')]
        elif line.startswith("Risk Factors:"):
            current = "risk_factors"
            details["risk_factors"] = [x.strip() for x in line.replace("Risk Factors:", "").split(',')]
        elif line.startswith("Prognosis:"):
            current = "prognosis"
            details["prognosis"] = line.replace("Prognosis:", "").strip()
        elif line.startswith("Treatments:"):
            current = "treatments"
            details["treatments"] = []
        elif current == "treatments" and line.strip():
            details["treatments"].append(line.strip())
    
    print("🧾 Structured Details:", details)
    return details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        image_data = file.read()
        label, name, confidence = classify_image(image_data)
        details = get_description(name)
        return render_template('index.html', result={
            'label': name,
            'confidence': f"{confidence * 100:.2f}%",
            'details': details
        })

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)