from src.transform import *
from src.utils import *
from model.resnet_18 import *

from flask import Flask, render_template, request, jsonify
from io import BytesIO

app = Flask(__name__)

config = load_config("config.yaml")

model = ResNet18()
model.load_state_dict(torch.load(config["model_path"]))
model.eval()

mean, std = compute_mean_std("input", for_training=False)

transform = v2.Compose([
    v2.Resize((config["height"], config["width"])),
    v2.ToTensor(),
    v2.Normalize(mean, std)
])

classes = ["bike", "bottle", "chair", "cup", "fork", "knife", "plant", "shoe", "spoon", "t-shirt"]

def predict_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities, 3)

    print(f"Top 3 indices: {top3_indices[0].tolist()}")  # Debugging output
    print(f"Top 3 probabilities: {top3_probs[0].tolist()}")  # Debugging output

    # Ensure the indices are valid
    top3_classes = [classes[idx] for idx in top3_indices[0].tolist()]
    top3_probs = top3_probs[0].tolist()

    return list(zip(top3_classes, top3_probs))


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Get the image bytes
        image_bytes = file.read()
        
        # Get the predictions
        predictions = predict_image(image_bytes)
        
        return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)