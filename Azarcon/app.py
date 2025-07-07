import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import tensorflow.lite as tflite
import numpy as np
import os
from PIL import Image
import traceback
from dotenv import load_dotenv  # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TFLite model
MODEL_PATH = 'skin_cancer_model/model.tflite'
model_path = os.path.join(os.getcwd(), MODEL_PATH)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Initialize Google GenAI
api_key = os.getenv("GEMINI_API_KEY")  # Load API key from .env
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")
genai.configure(api_key=api_key)

# Initialize the correct model object
model = genai.GenerativeModel("gemini-1.5-flash-latest")

def get_disease_info(predicted_name):
    """Fetch disease description and remedies using Google GenAI."""
    try:
        prompt = f"Provide a brief description, sympotms, and possible remedies in 5-8 sentences for the disease: {predicted_name}."
        print(f"Sending prompt to GenAI: {prompt}")  # Log the prompt

        # Ensure 'model' is being used correctly
        response = model.generate_content(prompt)  

        print(f"GenAI response: {response.text}")  # Log the raw response

        # Return the response as a dictionary
        return {
            "description": response.text if response.text else "No information available."
        }
    except Exception as e:
        import traceback
        print("Error fetching disease info:")
        traceback.print_exc()  # Log the full traceback
        return {
            "error": "Error fetching disease information."
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            print(f"File saved at: {filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Preprocess the image and make a prediction
        image = Image.open(filepath).resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]  # Get the first prediction
        
        # Define the labels and names
        labels = [
            {"label": "akiec", "name": "Actinic Keratoses and Intraepithelial Carcinomae"},
            {"label": "bcc", "name": "Basal Cell Carcinoma"},
            {"label": "bkl", "name": "Benign Keratosis"},
            {"label": "df", "name": "Dermatofibroma"},
            {"label": "mel", "name": "Melanoma"},
            {"label": "nv", "name": "Melanocytic Nevi"},
            {"label": "vasc", "name": "Vascular Skin Lesions"}
        ]
        
        # Find the predicted class
        predicted_index = np.argmax(prediction)  # Get the index of the highest probability
        predicted_label = labels[predicted_index]  # Get the corresponding label and name

        # Fetch disease information from Google GenAI
        disease_info = get_disease_info(predicted_label['name'])

        # Return the predicted_name and disease info in the response
        return jsonify({
            'message': 'File uploaded successfully',
            'filepath': filepath,
            'prediction': prediction.tolist(),
            'predicted_label': predicted_label['label'],
            'predicted_name': predicted_label['name'],
            'disease_info': disease_info  # Already a dictionary, jsonify will handle it
        })

if __name__ == '__main__':
    app.run(debug=True)