from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import cv2

# Load model
model_name = "prithivMLmods/Recycling-Net-11"
processor = AutoImageProcessor.from_pretrained(model_name)
model = SiglipForImageClassification.from_pretrained(model_name)

# Get official model classes
id2label = model.config.id2label
valid_labels = set(label.lower() for label in id2label.values())

# Mapping to two categories
recyclable_labels = {"cardboard", "glass", "metal", "paper", "plastic", "can", "carton"}
non_recyclable_labels = {"food waste", "trash", "organic", "garbage"}

def classify_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

    pred_idx = torch.argmax(probs).item()
    pred_label = id2label[pred_idx].lower()
    confidence = probs[pred_idx].item()

    # Only use known labels
    if pred_label not in valid_labels:
        return "Unknown", confidence

    # Main classification
    if pred_label in recyclable_labels:
        return "Recyclable", confidence
    else:
        return "Non-Recyclable", confidence

# Webcam loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, confidence = classify_frame(frame)
    text = f"{label} ({confidence*100:.1f}%)"
    color = (0, 255, 0) if "Recyclable" in label else (0, 0, 255)

    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Live Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
