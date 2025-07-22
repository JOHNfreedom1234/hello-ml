import json
import os
import logging
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai


load_dotenv('.env')
api_key = os.getenv('GENAI_API_KEY')
print(api_key)
try: 
    if not api_key:
        raise ValueError("Missing GENAI_API_KEY in .env file")
except ValueError as e:
    logging.error(f"Environment variable error: {e}")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Google GenAI client
client = genai.Client(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def model():
    """Loads the TensorFlow Lite model."""
    try:
        model_path_full = os.path.join(BASE_DIR, "skin_cancer_model", "model.tflite")
        classifier = tf.lite.Interpreter(model_path=model_path_full)
        classifier.allocate_tensors()
        input_details = classifier.get_input_details()
        output_details = classifier.get_output_details()
        return classifier, input_details, output_details
    except Exception as e:
        log.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model")
    
# Load model once at startup
classifier, input_details, output_details = model()

def load_labels():
    label_loc_full = os.path.join(BASE_DIR, "skin_cancer_model", "labels.json")
    if os.path.exists(label_loc_full):
        try:
            with open(label_loc_full, "r") as f:
                cancer_data = json.load(f)
            return [entry.get("name", "Unknown") for entry in cancer_data]
        except Exception as e:
            log.error(f"Error loading labels from {label_loc_full}: {e}")
            return []
    else:
        log.error(f"labels.json file not found at {label_loc_full}")
        return []
    
def preprocessing(image_path):
    try:
       with Image.open(image_path) as img:
           img = img.convert("RGB")
           img = img.resize((224, 224))
           img_array = np.array(img) / 255.0 
           img_array = np.expand_dims(img_array, axis=0)
           return img_array
    except UnidentifiedImageError:
        log.error("Unidentified image error")
        return None
    except Exception as e:
        log.error(f"Error preprocessing image: {e}")
        return None

def cancer(disease_name): 
    """Uses Gemini to provide information about the skin cancer"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            prompt=f"Provide information about {disease_name} skin cancer.",
            max_output_tokens=750
        )
        return response.candidates[0].content.parts[0].text.strip() if response.candidates else "No information available."
    except Exception as e:
        log.error(f"Error generating text with Gemini: {e}")
        return "Error generating information."

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    log.info("Received upload request")

    if 'file' not in request.files:
        log.warning("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        log.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(BASE_DIR, secure_filename(file.filename))
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        file.save(image_path)
        log.info(f"Image saved to {image_path}")

        img = None 
        try: 
            img = preprocessing(image_path).astype(np.float32)
        except Exception as e:
            log.error(f"Error processing image: {e}")
            return jsonify({"error": "Image processing failed"}), 500
        
        if classifier is None or input_details is None or output_details is None:
            log.error("Model not loaded properly")
            return jsonify({"error": "Model not loaded"}), 500

        # Inference Time
        classifier.set_tensor(input_details[0]['index'], img)
        classifier.invoke()
        output_data = classifier.get_tensor(output_details[0]['index'])
        log.info(f"Model output: {output_data}")

        # Post-process the output
        predicted_class = np.argmax(output_data)
        confidence = output_data[0][predicted_class]
        log.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        # Get the label for the predicted class
        labels = load_labels()
        if labels:
            predicted_label = labels[predicted_class]
        else:
            predicted_label = "Unknown"
        
        details = cancer(predicted_label)

        response = {
            "predicted_class": predicted_label,
            "details": details
        }

        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                log.error(f"Error removing image file: {e}")
        log.info("Image file removed after processing")

        return jsonify(response)

    except ValueError as e:
        log.error(f"Value error: {e}")
        return jsonify({"error": "Invalid input"}), 400
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

