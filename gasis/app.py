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
LABELS_JSON_PATH = r"C:\Users\LENOVO\OneDrive - wvsu.edu.ph\GitHub\hello-ml\gasis\skin_cancer_model\labels.json"

# Try to load from JSON first, then fallback to .txt
if os.path.exists(LABELS_JSON_PATH):
    print("📋 Loading labels from JSON file...")
    with open(LABELS_JSON_PATH, 'r') as file:
        label_data = json.load(file)
    
    # Handle different JSON structures
    if isinstance(label_data, dict):
        # If it's a dictionary mapping indices to names
        labels = []
        label_names = {}
        for key in sorted(label_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            short_name = key
            full_name = label_data[key]
            labels.append(short_name)
            label_names[short_name] = full_name
    elif isinstance(label_data, list):
        # If it's a list of dictionaries or just strings
        labels = []
        label_names = {}
        for i, item in enumerate(label_data):
            if isinstance(item, dict):
                short_name = item.get('short_name', item.get('class', f'class_{i}'))
                full_name = item.get('full_name', item.get('name', short_name))
            else:
                short_name = str(item)
                full_name = str(item)
            labels.append(short_name)
            label_names[short_name] = full_name
    else:
        raise ValueError("Unsupported JSON structure in labels file")
    
    print(f"✅ Loaded {len(labels)} labels from JSON")
    print("🏷️ Label mapping preview:")
    for i, (short, full) in enumerate(list(label_names.items())[:3]):
        print(f"   {i}: {short} -> {full}")
    
elif os.path.exists(LABELS_PATH):
    print("📋 Loading labels from text file...")
    with open(LABELS_PATH, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    label_names = {label: label for label in labels}
    print(f"✅ Loaded {len(labels)} labels from text file")
else:
    raise FileNotFoundError(f"Neither labels.json nor labels.txt found. Checked paths:\n- {LABELS_JSON_PATH}\n- {LABELS_PATH}")

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
        f"[Provide a comprehensive description of what this condition is in 3-5 sentences.]\n\n"
        f"**Causes:**\n"
        f"[List the main causes]\n\n"
        f"**Risk Factors:**\n"
        f"[List the risk factors]\n\n"
        f"**Prognosis:**\n"
        f"[Explain the typical prognosis and outlook]\n\n"
        f"**Treatments:**\n"
        f"[List available treatment options]\n\n"
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
    app.run(debug=True)