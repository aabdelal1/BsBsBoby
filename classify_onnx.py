import sys
import json
import os
import numpy as np
from PIL import Image
import onnxruntime as ort

def preprocess_image(image_path, size=200):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = img_data.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_data = (img_data - mean) / std
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No image path provided"}))
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(json.dumps({"success": False, "error": f"Image file not found: {image_path}"}))
        return

    try:
        labels_path = os.path.join("models", "labels.json")
        model_path = os.path.join("models", "pet_classifier_model.onnx")
        
        # Load labels
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            
        # Preprocess
        input_data = preprocess_image(image_path)
        
        # Run inference
        session = ort.InferenceSession(model_path)
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})
        logits = outputs[0][0]
        
        # Softmax probabilities
        probs = softmax(logits)
        pred_idx = np.argmax(probs)
        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx])
        
        print(json.dumps({
            "success": True,
            "prediction": pred_label,
            "confidence": confidence
        }))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    main()
