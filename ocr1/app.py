from flask import Flask, render_template, request, jsonify
import cv2
import pytesseract
import numpy as np
import base64
from pytesseract import Output

# Configure Tesseract path (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

def perform_ocr(image):
    """
    Perform Optical Character Recognition (OCR) on an image and return the recognized text.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        str: Recognized text from the image.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Increase intensity by applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        text = pytesseract.image_to_string(enhanced, lang='eng')
        return text
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and perform OCR."""
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded."})

    # Read the uploaded image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform OCR
    text = perform_ocr(image)
    return jsonify({"text": text})

@app.route('/capture', methods=['POST'])
def capture():
    """Handle image capture from the camera and perform OCR."""
    data_url = request.json.get('image')
    if not data_url:
        return jsonify({"error": "No image data provided."})

    # Decode the base64 image
    header, encoded = data_url.split(',', 1)
    image_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Perform OCR
    text = perform_ocr(image)
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(debug=True)
