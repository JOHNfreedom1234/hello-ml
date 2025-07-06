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

# Flexible model path setup for both local and Render deployment
def get_model_paths():
    """Get model paths that work both locally and on Render"""
    # Check if we're running on Render (or any Linux environment)
    if os.name != 'nt':  # Not Windows
        # For Render deployment - models should be in your project directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'skin_cancer_model')
        
        # Alternative locations to check
        possible_dirs = [
            model_dir,
            os.path.join(base_dir, 'model'),
            os.path.join(base_dir, 'models'),
            base_dir  # Check root directory
        ]
        
        for directory in possible_dirs:
            model_path = os.path.join(directory, 'model.tflite')
            labels_txt_path = os.path.join(directory, 'labels.txt')
            labels_json_path = os.path.join(directory, 'labels.json')
            
            if os.path.exists(model_path):
                print(f"📍 Found model in: {directory}")
                return model_path, labels_txt_path, labels_json_path
        
        # If not found, use relative paths (Render will look in project root)
        return './model.tflite', './labels.txt', './labels.json'
    
    else:
        # Local Windows development paths
        local_path = r"C:\Users\LENOVO\OneDrive - wvsu.edu.ph\GitHub\hello-ml\gasis\skin_cancer_model"
        return (
            os.path.join(local_path, "model.tflite"),
            os.path.join(local_path, "labels.txt"),
            os.path.join(local_path, "labels.json")
        )

# Get appropriate paths
MODEL_PATH, LABELS_PATH, LABELS_JSON_PATH = get_model_paths()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

print(f"🔍 Looking for model at: {MODEL_PATH}")
print(f"🔍 Looking for labels at: {LABELS_PATH} or {LABELS_JSON_PATH}")

# Load TFLite model
if not os.path.exists(MODEL_PATH):
    # List available files for debugging
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"❌ Model not found at: {MODEL_PATH}")
    print(f"📂 Current directory: {current_dir}")
    print(f"📂 Files in current directory: {os.listdir(current_dir)}")
    
    # Check for model files in current directory
    for file in os.listdir(current_dir):
        if file.endswith('.tflite'):
            print(f"🔍 Found .tflite file: {file}")
            MODEL_PATH = os.path.join(current_dir, file)
            break
    else:
        raise FileNotFoundError(f"Model not found. Checked: {MODEL_PATH}")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Load label names with flexible path handling
def load_labels():
    """Load labels from JSON or TXT file with flexible path handling"""
    
    # Try JSON first
    json_paths_to_try = [
        LABELS_JSON_PATH,
        os.path.join(os.path.dirname(MODEL_PATH), 'labels.json'),
        './labels.json',
        'labels.json'
    ]
    
    for json_path in json_paths_to_try:
        if os.path.exists(json_path):
            print(f"📋 Loading labels from JSON file: {json_path}")
            try:
                with open(json_path, 'r') as file:
                    label_data = json.load(file)
                
                labels = []
                label_names = {}
                
                # Handle different JSON structures
                if isinstance(label_data, dict):
                    for key in sorted(label_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
                        short_name = key
                        full_name = label_data[key]
                        labels.append(short_name)
                        label_names[short_name] = full_name
                elif isinstance(label_data, list):
                    for i, item in enumerate(label_data):
                        if isinstance(item, dict):
                            short_name = item.get('short_name', item.get('class', f'class_{i}'))
                            full_name = item.get('full_name', item.get('name', short_name))
                        else:
                            short_name = str(item)
                            full_name = str(item)
                        labels.append(short_name)
                        label_names[short_name] = full_name
                
                print(f"✅ Loaded {len(labels)} labels from JSON")
                return labels, label_names
                
            except Exception as e:
                print(f"❌ Error loading JSON labels from {json_path}: {e}")
                continue
    
    # Try TXT files
    txt_paths_to_try = [
        LABELS_PATH,
        os.path.join(os.path.dirname(MODEL_PATH), 'labels.txt'),
        './labels.txt',
        'labels.txt'
    ]
    
    for txt_path in txt_paths_to_try:
        if os.path.exists(txt_path):
            print(f"📋 Loading labels from text file: {txt_path}")
            try:
                with open(txt_path, 'r') as file:
                    labels = [line.strip() for line in file.readlines()]
                label_names = {label: label for label in labels}
                print(f"✅ Loaded {len(labels)} labels from text file")
                return labels, label_names
            except Exception as e:
                print(f"❌ Error loading text labels from {txt_path}: {e}")
                continue
    
    # Fallback - create default labels
    print("⚠️ No labels file found, using default labels")
    default_labels = [
        'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
    ]
    default_names = {
        'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis-like Lesions',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesions'
    }
    return default_labels, default_names

# Load labels
labels, label_names = load_labels()

# Print label mapping preview
print("🏷️ Label mapping preview:")
for i, (short, full) in enumerate(list(label_names.items())[:3]):
    print(f"   {i}: {short} -> {full}")

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
    
    # Get the short label and corresponding full name
    short_label = labels[pred_index]
    full_name = label_names.get(short_label, short_label)
    confidence = float(output_data[pred_index])
    
    print(f"🔍 Classification result: {short_label} -> {full_name} ({confidence:.4f})")
    return short_label, full_name, confidence

# Improved Gemini response handling
def get_description(label_name):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = (
        f"Provide detailed information about the skin condition '{label_name}' in the following format:\n\n"
        f"**Description:**\n"
        f"[Provide a comprehensive description of what this condition is in 3-5 sentences and remove any asterisks for bold texts.]\n\n"
        f"**Causes:**\n"
        f"[List the main causes and remove any asterisks for bold texts]\n\n"
        f"**Risk Factors:**\n"
        f"[List the risk factors and remove any asterisks for bold texts]\n\n"
        f"**Prognosis:**\n"
        f"[Explain the typical prognosis and outlook and remove any asterisks for bold texts]\n\n"
        f"**Treatments:**\n"
        f"[List available treatment options and remove any asterisks for bold texts]\n\n"
        f"Please be thorough and medically accurate."
    )

    try:
        response = model.generate_content(prompt)
        print("🧠 Gemini Raw Response:")
        print(response.text)

        # Initialize structured output with fallback
        details = {
            "description": "No description available",
            "causes": ["Information not available"],
            "risk_factors": ["Information not available"],
            "prognosis": "Information not available",
            "treatments": ["Information not available"]
        }

        # Improved parsing logic
        response_text = response.text.strip()
        
        # Split into sections based on bold headers
        sections = {}
        current_section = None
        current_content = []
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check for section headers (with or without asterisks)
            if any(header in line.lower() for header in ['description:', '**description**', 'description']):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'description'
                current_content = []
                # Add content after the header if it exists on the same line
                content_after_header = line.split(':', 1)[-1].strip()
                if content_after_header and not content_after_header.startswith('*'):
                    current_content.append(content_after_header)
            elif any(header in line.lower() for header in ['causes:', '**causes**', 'causes']):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'causes'
                current_content = []
            elif any(header in line.lower() for header in ['risk factors:', '**risk factors**', 'risk factors']):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'risk_factors'
                current_content = []
            elif any(header in line.lower() for header in ['prognosis:', '**prognosis**', 'prognosis']):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'prognosis'
                current_content = []
            elif any(header in line.lower() for header in ['treatments:', '**treatments**', 'treatments']):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'treatments'
                current_content = []
            elif line and current_section:
                # Add content to current section
                current_content.append(line)
        
        # Don't forget the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Process the sections into the details dictionary
        if 'description' in sections and sections['description']:
            details['description'] = sections['description']
        
        if 'causes' in sections and sections['causes']:
            # Split by lines, bullets, or commas and clean up
            causes_text = sections['causes']
            causes = []
            for line in causes_text.split('\n'):
                line = line.strip()
                if line:
                    # Remove bullet points and numbers
                    line = line.lstrip('•-*123456789. ')
                    if line:
                        causes.append(line)
            if causes:
                details['causes'] = causes
        
        if 'risk_factors' in sections and sections['risk_factors']:
            risk_factors_text = sections['risk_factors']
            risk_factors = []
            for line in risk_factors_text.split('\n'):
                line = line.strip()
                if line:
                    line = line.lstrip('•-*123456789. ')
                    if line:
                        risk_factors.append(line)
            if risk_factors:
                details['risk_factors'] = risk_factors
        
        if 'prognosis' in sections and sections['prognosis']:
            details['prognosis'] = sections['prognosis']
        
        if 'treatments' in sections and sections['treatments']:
            treatments_text = sections['treatments']
            treatments = []
            for line in treatments_text.split('\n'):
                line = line.strip()
                if line:
                    line = line.lstrip('•-*123456789. ')
                    if line:
                        treatments.append(line)
            if treatments:
                details['treatments'] = treatments
        
        # Fallback: if we couldn't parse properly, use the entire response as description
        if details['description'] == "No description available" and response_text:
            details['description'] = response_text
        
        print("🧾 Structured Details:", details)
        return details
        
    except Exception as e:
        print(f"❌ Error getting description from Gemini: {e}")
        return {
            "description": f"Error retrieving information about {label_name}",
            "causes": ["Information unavailable due to API error"],
            "risk_factors": ["Information unavailable due to API error"],
            "prognosis": "Information unavailable due to API error",
            "treatments": ["Information unavailable due to API error"]
        }

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
        try:
            image_data = file.read()
            label, name, confidence = classify_image(image_data)
            details = get_description(name)
            return render_template('index.html', result={
                'label': name,
                'confidence': f"{confidence * 100:.2f}%",
                'details': details
            })
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return jsonify({'error': 'Error processing image'}), 500

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    # Use environment variable for port (Render requirement)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)