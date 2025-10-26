from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64
import numpy as np
import sys
sys.path.append('..')

from models.model import create_model
from utils.gradcam import GradCAM
from utils.data_loader import get_transforms

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(num_classes=4, device = device)
checkpoint = torch.load('../models/saved_models/best_brain_tumor_model.pth',
                       map_location=device,
                       weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

_, transform = get_transforms(image_size=224)

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

target_layer = model.backbone.layer4[-1]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    img_bytes = file.read()
    original_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    input_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_class = probs.argmax()
    confidence = probs[pred_class] * 100

    gradcam = GradCAM(model, target_layer)
    cam, _ = gradcam.generate_cam(input_tensor)
    overlayed = gradcam.visualize(np.array(original_image), cam, alpha = 0.4)

    def img_to_base64(img_array):
        img = Image.fromarray(img_array.astype('uint8'))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    original_b64 = img_to_base64(np.array(original_image))
    overlayed_b64 = img_to_base64(overlayed)

    response = {
        'prediction': classes[pred_class],
        'confidence': float(confidence),
        'probabilities': {
            classes[i]: float(probs[i] * 100) for i in range(len(classes))
        },
        'original_image': original_b64,
        'gradcam_image': overlayed_b64
    }

    return jsonify(response)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("BRAIN TUMOR DETECTION DASHBOARD")
    print("="*60)
    print(f"Device: {device}")
    print(f"\n Server starting on http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)

