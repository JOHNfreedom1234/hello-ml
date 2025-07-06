import os
import json
import numpy as np
import tensorflow.lite as tflite
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set Flask config for uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model and labels paths
MODEL_PATH = os.path.abspath("models/model_unquant.tflite")
LABELS_JSON_PATH = os.path.join(os.getcwd(), "models", "labels.json")

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels from labels.json
with open(LABELS_JSON_PATH, 'r') as f:
    label_data = json.load(f)

# Create a dictionary mapping index (as integer) to full name
index_to_name = {int(item["label"]): item["name"] for item in label_data}

def preprocess_image(image_path):
    """Preprocesses image to match model input requirements."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))  # Resize to model input size
    image = np.array(image, dtype=np.float32) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image_path):
    """Runs inference on an image and returns the predicted class."""
    try:
        image = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Ensure correct prediction index
        predicted_index = int(np.argmax(output_data))  

        # Debugging output
        print("Model Output:", output_data)
        print("Predicted Index:", predicted_index)

        # Return full name if found, else "Unknown"
        return index_to_name.get(predicted_index, "Unknown")
    except Exception as e:
        return f"Prediction Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handles image upload and returns model prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        prediction = predict(file_path)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
