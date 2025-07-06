import tensorflow.lite as tflite

import os
MODEL_PATH = os.path.abspath("model.tflite")


try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
