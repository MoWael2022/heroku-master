from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import easyocr
from io import BytesIO
from PIL import Image
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])


# Function to download image from URL with preprocessing
def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            sharp = cv2.filter2D(blur, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
            resized_img = cv2.resize(sharp, (300, 300))
            return resized_img
        else:
            return None
    except Exception as e:
        return None


# Function to detect text using EasyOCR
def detect_text_easyocr(image):
    try:
        result = reader.readtext(image)
        return result
    except Exception as e:
        return []


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' in request.files:
        file = request.files['file']
        image = Image.open(file.stream)
    elif 'url' in request.json:
        url = request.json['url']
        image = download_image(url)
        if image is None:
            return jsonify({'error': 'Failed to download image from URL'}), 400
    else:
        return jsonify({'error': 'No image file or URL provided'}), 400

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    text_boxes = detect_text_easyocr(image_cv)

    if not text_boxes:
        return jsonify({'detected_text': 'No text detected'}), 200

    # Find the bounding box with the largest area
    max_area = 0
    largest_text = ""
    for (bbox, text, _) in text_boxes:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        width = int(np.linalg.norm(np.array(top_right) - np.array(top_left)))
        height = int(np.linalg.norm(np.array(top_right) - np.array(bottom_right)))
        area = width * height
        if area > max_area:
            max_area = area
            largest_text = text

    return jsonify({'detected_text': largest_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)