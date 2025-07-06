from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key. Please check your .env file.")

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

def preprocess_image(image_path, target_shape):
    """Resize and normalize the image for prediction."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((target_shape[1], target_shape[2]))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def read_labels_txt(path):
    """Read label mapping from a text file."""
    with open(path, 'r') as f:
        lines = f.readlines()
    return {str(idx): line.strip() for idx, line in enumerate(lines)}

def read_labels_json(path):
    """Read label metadata from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['label']: item['name'] for item in data}

def get_disease_details(label):
    """Use Gemini AI to generate disease information."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Explain the disease '{label}' in detail. "
            f"Include symptoms, causes, diagnosis, and treatment options."
        )
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No info available."
    except Exception as err:
        print(f"Gemini API error: {err}")
        return "Could not retrieve disease details."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return render_template('error.html', error="No file was uploaded."), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('error.html', error="No file selected."), 400

    # Save file
    save_path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(save_path)

    # Load TFLite model
    model_path = "model/model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Model input/output
    input_info = interpreter.get_input_details()
    output_info = interpreter.get_output_details()
    input_shape = input_info[0]['shape']

    # Prepare image
    image_tensor = preprocess_image(save_path, input_shape)

    try:
        interpreter.set_tensor(input_info[0]['index'], image_tensor)
        interpreter.invoke()
        prediction_output = interpreter.get_tensor(output_info[0]['index'])
        scores = np.squeeze(prediction_output)
    except Exception as e:
        return render_template('error.html', error=f"Inference failed: {e}"), 500

    # Load labels
    txt_labels = read_labels_txt("model/labels.txt")
    json_labels = read_labels_json("model/labels.json")

    # Match predictions
    results = []
    for idx, score in enumerate(scores):
        raw_label = txt_labels.get(str(idx), f"Unknown {idx}")
        display_name = json_labels.get(raw_label, "Unknown")
        results.append({
            'label': display_name,
            'score': float(score)
        })

    top_result = max(results, key=lambda x: x['score'])
    summary = f"Prediction: {top_result['label']} (Confidence: {top_result['score']:.2f})"

    # Get AI-generated disease info
    details = get_disease_details(top_result['label'])

    return render_template('results.html', results=results, classification=summary, disease_info=details)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
