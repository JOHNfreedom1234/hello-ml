import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path='skin_cancer_model/model.tflite')
    interpreter.allocate_tensors()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info(f"Input details: {input_details}")
logger.info(f"Output details: {output_details}")

# Load labels
try:
    with open('skin_cancer_model/labels.json', 'r') as f:
        labels = json.load(f)
    logger.info("Labels loaded successfully")
except Exception as e:
    logger.error(f"Error loading labels: {str(e)}")
    raise

def preprocess_image(image_path):
    logger.debug(f"Preprocessing image: {image_path}")
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        logger.debug(f"Image size before resize: {img.size}")
        img = img.resize((224, 224))
        logger.debug(f"Image size after resize: {img.size}")
        img = np.array(img) / 255.0
        logger.debug(f"Image array shape: {img.shape}")
        img = np.expand_dims(img, axis=0)
        logger.debug(f"Final preprocessed shape: {img.shape}")
        return img.astype(np.float32)
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def predict_image(image_path):
    logger.info(f"Starting prediction for image: {image_path}")
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        logger.debug(f"Preprocessed image shape: {processed_image.shape}")
        
        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        logger.debug("Input tensor set successfully")
        
        # Run the inference
        interpreter.invoke()
        logger.debug("Model inference completed")
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        logger.debug(f"Raw output: {output_data}")
        
        # Get the prediction
        prediction_idx = np.argmax(output_data[0])
        confidence = float(output_data[0][prediction_idx])
        logger.debug(f"Prediction index: {prediction_idx}, Confidence: {confidence}")
        
        # Get the label information
        label_info = labels[prediction_idx]
        logger.info(f"Predicted class: {label_info['name']}, Confidence: {confidence * 100}%")
        
        return {
            'class': f"{label_info['name']} ({label_info['label']})",
            'confidence': confidence * 100
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'error': 'No file selected'})
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to: {filepath}")
            file.save(filepath)
            
            try:
                result = predict_image(filepath)
                logger.info(f"Prediction result: {result}")
                os.remove(filepath)  # Clean up the uploaded file
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                os.remove(filepath)  # Clean up the uploaded file
                return jsonify({'error': str(e)})
        except Exception as e:
            logger.error(f"Error handling file: {str(e)}")
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 