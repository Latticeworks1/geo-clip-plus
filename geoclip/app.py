from flask import Flask, request, jsonify, send_file
from io import BytesIO
from geoclip_service import GeoCLIPService  # Assuming GeoCLIPService is in geoclip_service.py

app = Flask(__name__)
service = GeoCLIPService()  # Initialize the service with default settings

@app.route('/')
def home():
    return "Welcome to the GeoCLIP Plus Service API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    top_k = int(request.form.get('top_k', 5))
    
    # Predict GPS coordinates
    predictions = service.predict(image_file, top_k=top_k)
    
    return jsonify(predictions)

@app.route('/heatmap', methods=['POST'])
def heatmap():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    top_k = int(request.form.get('top_k', 5))
    
    # Predict GPS coordinates
    predictions = service.predict(image_file, top_k=top_k)
    
    # Generate heatmap
    heatmap = service.create_heatmap(predictions)
    
    # Save the heatmap to a BytesIO object
    heatmap_data = BytesIO()
    heatmap.save(heatmap_data, format='html')
    heatmap_data.seek(0)

    return send_file(heatmap_data, mimetype='text/html', as_attachment=True, attachment_filename='heatmap.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
