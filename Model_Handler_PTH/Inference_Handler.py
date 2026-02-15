import torch
import os
from Create_Model import Model_Architecture

from Model_Handler_PTH import CV_Processor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOADED_MODELS = {}

def get_model_structure(model_id: str):
    
    if model_id == "ModernCNN":
        return Model_Architecture.ModernCNN()
    
    
    raise ValueError(f"Unknown model id: {model_id}")

def load_model(model_path: str, model_id: str):
    
    global LOADED_MODELS
    
    if model_id in LOADED_MODELS:
        return LOADED_MODELS[model_id]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from disk: {model_id}...")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)    
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model = get_model_structure(model_id)
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state_dict, trying full model load: {e}")
        print(f"Direct load failed, trying to strip 'module.' prefix: {e}")
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
    model.to(DEVICE)
    model.eval()
    
    LOADED_MODELS[model_id] = model
    return model

def predict_digit_real(image_path: str, model_path: str, model_id: str):
    try:
        processed_img = CV_Processor.preprocess_image_for_mnist(image_path)
        img_tensor = torch.tensor(processed_img, dtype=torch.float32).to(DEVICE)
        
        if img_tensor.shape[-1] == 3: # اگر رنگی بود
            img_tensor = img_tensor.mean(dim=2, keepdim=True)
        
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        model = load_model(model_path, model_id)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            digit = predicted_class.item()
            conf_score = confidence.item() * 100

        return {
            "digit": digit,
            "confidence": f"{conf_score:.2f}%",
            "details": f"Model: {model_id} (Real Inference)"
        }

    except Exception as e:
        print(f"Prediction Error for {model_id}: {e}")
        return {
            "digit": -1,
            "confidence": "0%",
            "details": f"Error: {str(e)}"
        }