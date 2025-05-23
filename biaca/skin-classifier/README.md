# Flask TensorFlow Lite Image Upload Application

This project is a simple Flask application that allows users to upload images and processes them using a TensorFlow Lite model. 

## Project Structure

```
flask-app
├── models          # Directory for TensorFlow Lite model files
├── uploads         # Directory for storing uploaded images
├── app.py          # Main entry point of the Flask application
├── requirements.txt # Lists the dependencies required for the project
└── README.md       # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd flask-app
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Add your TensorFlow Lite model:**
   Place your `.tflite` model file in the `models` directory.

5. **Run the application:**
   ```
   python app.py
   ```

## Usage

- Navigate to `http://localhost:5000` in your web browser.
- Use the provided interface to upload an image.
- The application will process the image using the TensorFlow Lite model and return the results.

## License

This project is licensed under the MIT License.