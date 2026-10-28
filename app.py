import os
import io
import torch
import timm #[cite: 3]
from flask import Flask, request, render_template, jsonify
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #[cite: 3]

# Define the exact 8 classes from your training data[cite: 3]
CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'] #[cite: 3]

# Load Model[cite: 3]
model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False, num_classes=8) #[cite: 3]
model.load_state_dict(torch.load('best_blood_group_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Validation transforms from your training script[cite: 3]
transform = transforms.Compose([
    transforms.Resize((224, 224)), #[cite: 3]
    transforms.ToTensor(), #[cite: 3]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #[cite: 3]
])

def predict_image(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            return CLASSES[predicted.item()]
    except Exception as e:
        return str(e)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    img_bytes = file.read()
    prediction = predict_image(img_bytes)
    return jsonify({'blood_group': prediction})

@app.route('/esp32_upload', methods=['POST'])
def esp32_upload():
    # ESP32 will send raw image bytes as binary payload
    img_bytes = request.data
    if not img_bytes:
        return "No image data received", 400
    
    prediction = predict_image(img_bytes)
    # Save the prediction to a temporary file or global variable so the web UI can poll it
    with open('latest_esp32_result.txt', 'w') as f:
        f.write(prediction)
        
    return "Prediction successful", 200

@app.route('/get_esp32_result', methods=['GET'])
def get_esp32_result():
    if os.path.exists('latest_esp32_result.txt'):
        with open('latest_esp32_result.txt', 'r') as f:
            return jsonify({'blood_group': f.read().strip()})
    return jsonify({'blood_group': 'Waiting for sensor...'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)