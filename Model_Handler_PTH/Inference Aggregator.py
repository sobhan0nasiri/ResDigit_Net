import torch
import torch.nn.functional as F
from Create_Model.Model_Architecture.Model_ModernCNN import ModernCNN

class DigitInferenceAggregator:
    def __init__(self, device=None):

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Aggregator initialized on: {self.device}")

    def add_model(self, model_name, model_arch, weights_path):

        model = model_arch.to(self.device)

        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            self.models[model_name] = model
            print(f"✅ Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model '{model_name}': {e}")

    def predict(self, image_tensor, selected_models=None):

        if not self.models:
            raise ValueError("هیچ مدلی در سیستم بارگذاری نشده است!")

        names_to_use = selected_models if selected_models else list(self.models.keys())
        
        all_probs = []

        with torch.no_grad():
            
            if image_tensor.ndimension() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)

            for name in names_to_use:
                if name in self.models:
                    logits = self.models[name](image_tensor)

                    probs = F.softmax(logits, dim=1)
                    all_probs.append(probs)
                else:
                    print(f"⚠️ Warning: Model '{name}' not found in registry.")

        if not all_probs:
            return None

        combined_probs = torch.stack(all_probs).mean(dim=0)

        confidence, predicted_label = torch.max(combined_probs, 1)
        
        return {
            'label': predicted_label.item(),
            'confidence': confidence.item() * 100,
            'all_probs': combined_probs.cpu().numpy()
        }

aggregator = DigitInferenceAggregator()

aggregator.add_model("MainModel", ModernCNN(num_classes=10), 'modern_cnn_digits.pth')