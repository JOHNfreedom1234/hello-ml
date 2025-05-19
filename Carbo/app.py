from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import tensorflow as tf

# Disable oneDNN warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Match .env variable name
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Path configurations
MODEL_DIR = "skin_cancer_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# Verify model and labels exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {os.path.abspath(MODEL_PATH)}")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found at {os.path.abspath(LABELS_PATH)}")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(LABELS_PATH) as f:
    labels = json.load(f)
label_dict = {item["label"]: item["name"] for item in labels}

def get_medical_advice(diagnosis):
    """Get AI-generated medical advice using Gemini"""
    prompt = f"""As a dermatology AI assistant, provide short (2-3 sentence), clear advice 
    for someone who might have {diagnosis}. Include next steps and reassurance. 
    Use simple, non-alarming language."""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Please consult a dermatologist for professional medical advice."

def preprocess_image(image_path):
    """Process image for TFLite model input"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224])  # Update size if your model requires different
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img.numpy(), axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Save uploaded file
        upload_dir = os.path.join(app.root_path, 'static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Preprocess and predict
        input_data = preprocess_image(file_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Process results
        predicted_class = np.argmax(predictions[0])
        label = list(label_dict.keys())[predicted_class]
        diagnosis = label_dict[label]
        advice = get_medical_advice(diagnosis)

        return jsonify({
            'diagnosis': diagnosis,
            'label': label,
            'advice': advice,
            'image_url': f'/static/uploads/{file.filename}'
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Error processing image'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)