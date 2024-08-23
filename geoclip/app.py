from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
from PIL import Image
import torch
from geoclip.model.GeoCLIP import GeoCLIP
import folium
from folium.plugins import HeatMap

app = Flask(__name__)

# Initialize the GeoCLIP model
model = GeoCLIP(from_pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    top_k = int(request.form.get('top_k', 5))
    
    # Load and preprocess the image
    img = Image.open(image_file)
    img = model.image_encoder.preprocess_image(img).unsqueeze(0).to(device)
    
    # Predict GPS coordinates
    top_pred_gps, top_pred_prob = model.predict(img, top_k)
    
    predictions = [
        {"lat": float(gps[0]), "lon": float(gps[1]), "confidence": float(prob)}
        for gps, prob in zip(top_pred_gps, top_pred_prob)
    ]
    
    return jsonify(predictions)

@app.route('/heatmap', methods=['POST'])
def heatmap():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    top_k = int(request.form.get('top_k', 5))
    
    # Load and preprocess the image
    img = Image.open(image_file)
    img = model.image_encoder.preprocess_image(img).unsqueeze(0).to(device)
    
    # Predict GPS coordinates
    top_pred_gps, top_pred_prob = model.predict(img, top_k)
    
    # Generate heatmap
    predictions = [
        (float(gps[0]), float(gps[1]), float(prob))
        for gps, prob in zip(top_pred_gps, top_pred_prob)
    ]
    heatmap = create_heatmap(predictions)
    
    # Save the heatmap to a BytesIO object
    heatmap_data = BytesIO()
    heatmap.save(heatmap_data, format='png')
    heatmap_data.seek(0)
    return send_file(heatmap_data, mimetype='image/png', as_attachment=True, attachment_filename='heatmap.png')

def create_heatmap(predictions):
    coords, weights = zip(*[(pred[:2], pred[2]) for pred in predictions])
    norm_weights = [w / sum(weights) for w in weights]
    
    center = (sum(lat for lat, _ in coords) / len(coords), sum(lon for _, lon in coords) / len(coords))
    m = folium.Map(location=center, zoom_start=2)
    
    HeatMap(list(zip(coords, norm_weights)), gradient={0.0: '#932667', 1.0: '#fcfdbf'}).add_to(m)
    folium.Marker(
        location=coords[0],
        popup=f"Top Prediction: {coords[0]} with confidence {norm_weights[0]:.4f}",
        icon=folium.Icon(color='orange', icon='star')
    ).add_to(m)
    return m

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
