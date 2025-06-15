from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import cv2

# Load model and processor
model_name = "prithivMLmods/Recycling-Net-11"
processor = AutoImageProcessor.from_pretrained(model_name)
model = SiglipForImageClassification.from_pretrained(model_name)

# Labels
recyclable_labels = ["cardboard", "glass", "metal", "paper", "plastic", "can", "carton"]
non_recyclable_labels = ["food waste", "trash", "garbage", "organic"]
id2label = model.config.id2label

# Flask App
from flask import Flask, Response
app = Flask(__name__)

def classify_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    pred_idx = max(range(len(probs)), key=lambda i: probs[i])
    pred_label = id2label[pred_idx].lower()
    if any(word in pred_label for word in recyclable_labels):
        return "Recyclable", probs[pred_idx]
    else:
        return "Non-Recyclable", probs[pred_idx]

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        label, confidence = classify_frame(frame)
        text = f"{label} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if label == "Recyclable" else (0, 0, 255)
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head><title>Live Waste Classifier</title></head>
        <body>
            <h2>Live Waste Classifier (Recyclable / Non-Recyclable)</h2>
            <img src="/video_feed">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860)
