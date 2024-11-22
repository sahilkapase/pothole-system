from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json

app = Flask(__name__, static_folder='/')  # Serve static files from the 'assets' folder

# Model and folders
model = YOLO(r"best.pt")
UPLOAD_FOLDER = "./uploads"
RESULT_FOLDER = "./results"
STATIC_FOLDER = "./static"
STATS_FILE = "./stats.json"

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

    # Perform model prediction
    results = model.predict(image, conf=0.25)
    processed_data = []

    # Process detections
    for idx, detection in enumerate(results[0].boxes):  # Add indexing
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        area = calculate_area(polygon)
        cost = calculate_cost(area)
        processed_data.append({
            'index': idx + 1,  # Add the pothole index
            'polygon': polygon,
            'area_m2': area,
            'cost_rupees': cost
        })

        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the pothole index near the top-left corner of the bounding box in black
        label_position = (x1, y1 - 10) if y1 > 20 else (x1, y1 + 20)  # Adjust position if too close to the edge
        cv2.putText(image, f">{idx + 1}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
    return abs(area) / 1000  # Area in square meters (scaling factor adjusted)


def calculate_cost(area):
    """Calculate maintenance cost based on area in rupees."""
    cost_per_square_meter_in_rupees = 50  # Cost per square meter in rupees
    return area * cost_per_square_meter_in_rupees  # Return cost in rupees


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


# Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route('/model.html')
def upload_page():
    """Serve the image upload page."""
    return render_template("model.html")


@app.route('/statistics.html')
def statistics_page():
    """Serve the statistics page."""
    return render_template("statistics.html")


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image upload and detection."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    # Save uploaded image
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    # Process image and detect potholes
    processed_data, result_image_filename = process_image(image_path)
    if not processed_data:
        return jsonify({'error': "Error processing the image."}), 500

    # Update statistics
    update_statistics(processed_data)

    # Return JSON response
    result_image_url = f"/results/{result_image_filename}"
    return jsonify({'detections': processed_data, 'result_image': result_image_url})


@app.route('/results/<filename>')
def serve_result_image(filename):
    """Serve the result image."""
    return send_from_directory(RESULT_FOLDER, filename)


@app.route('/assets/<filename>')
def serve_asset(filename):
    """Serve files from the assets folder."""
    return send_from_directory('assets', filename)


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Return aggregated pothole statistics."""
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
    return jsonify(stats)


# Run app
if __name__ == "__main__":
    app.run(debug=True)
