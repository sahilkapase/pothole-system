from flask import Flask, request, jsonify, render_template, send_from_directory, abort
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
RESULT_FOLDER = os.environ.get('RESULT_FOLDER', './results')
STATIC_FOLDER = os.environ.get('STATIC_FOLDER', './static')
STATS_FILE = os.environ.get('STATS_FILE', './stats.json')

app = Flask(__name__, static_folder='/')
app.secret_key = SECRET_KEY

# Model and folders
model = YOLO(r"best.pt")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize statistics file
if not os.path.exists(STATS_FILE):
    with open(STATS_FILE, "w") as f:
        json.dump({"total_detections": 0, "area_data": [], "cost_data": []}, f)


def process_image(image_path):
    """Process the uploaded image and detect potholes."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not read the image."

    # Get the total image area
    total_image_area = image.shape[0] * image.shape[1]

    # Perform model prediction
    results = model.predict(image, conf=0.25)
    processed_data = []

    # Process detections
    for idx, detection in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        area = calculate_area(polygon)
        cost = calculate_cost(area)

        # Calculate percentage damage
        percentage_damage = (area * 100) / (total_image_area / 1000)

        processed_data.append({
            'index': idx + 1,
            'polygon': polygon,
            'area_m2': area,
            'cost_rupees': cost,
            'percentage_damage': percentage_damage
        })

        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_position = (x1, y1 - 10) if y1 > 20 else (x1, y1 + 20)
        cv2.putText(image, f"{idx + 1}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Save result image
    result_image_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_image_path, image)
    return processed_data, os.path.basename(result_image_path)


def calculate_area(polygon):
    """Calculate polygon area using the Shoelace formula."""
    if len(polygon) < 3:
        return 0
    area = 0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 1000


def calculate_cost(area):
    """Calculate maintenance cost based on area in rupees."""
    cost_per_square_meter_in_rupees = 50
    return area * cost_per_square_meter_in_rupees


def update_statistics(detections):
    """Update statistics with the new detections."""
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)

    for detection in detections:
        stats["total_detections"] += 1
        stats["area_data"].append(detection["area_m2"])
        stats["cost_data"].append(detection["cost_rupees"])

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/model.html')
def upload_page():
    return render_template("model.html")


@app.route('/statistics.html')
def statistics_page():
    return render_template("statistics.html")


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    image_file = request.files['image']
    filename = image_file.filename
    if not allowed_file(filename):
        return jsonify({'error': 'Invalid file type. Only .jpg, .jpeg, .png allowed.'}), 400

    image_file.seek(0, os.SEEK_END)
    file_size = image_file.tell()
    image_file.seek(0)
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large. Max size is 5MB.'}), 400

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    processed_data, result_image_filename = process_image(image_path)
    if not processed_data:
        return jsonify({'error': "Error processing the image."}), 500

    update_statistics(processed_data)

    result_image_url = f"/results/{result_image_filename}"
    return jsonify({'detections': processed_data, 'result_image': result_image_url})


@app.route('/api/statistics')
def get_statistics():
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
    return jsonify(stats)


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)
