import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class Model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load MLP LP-FT model
        self.model_path = os.path.join(
            os.path.dirname(__file__), 
            'mlp_lpft_fold1_fp16.pt'
        )
        print(f"Loading MLP LP-FT model from {self.model_path}")
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()
        
        self.use_fp16 = self.device.type == 'cuda'
        
        # Base Transform
        self.base_resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
    def load(self):
        """Codabench requires this method, even if loading is done in __init__."""
        pass

    def format_predictions(self, SPEI_30d: float, SPEI_1y: float, SPEI_2y: float) -> dict:
        return {
            "SPEI_30d": {"mu": float(SPEI_30d), "sigma": 0.25},
            "SPEI_1y": {"mu": float(SPEI_1y), "sigma": 0.30},
            "SPEI_2y": {"mu": float(SPEI_2y), "sigma": 0.22}
        }

    def predict(self, x) -> dict:
        if isinstance(x, list):
            x = x[0]
            
        img_path = x.get('file_path')
        if not img_path or not os.path.exists(img_path):
            return self.format_predictions(0.0, 0.0, 0.0)
            
        try:
            image = Image.open(img_path).convert('RGB')
            base_img = self.base_resize(image)
            tensor = self.normalize(self.to_tensor(base_img)).unsqueeze(0).to(self.device)
            
            if self.use_fp16:
                tensor = tensor.half()
                
            with torch.no_grad():
                preds = self.model(tensor)  # (1, 3)
                
            return self.format_predictions(
                SPEI_30d=preds[0, 0].item(), 
                SPEI_1y=preds[0, 1].item(), 
                SPEI_2y=preds[0, 2].item()
            )

        except Exception as e:
            print(f"Error predicting {img_path}: {e}")
            return self.format_predictions(0.0, 0.0, 0.0)
