import torch
from PIL import Image
from typing import Union, List, Tuple, Optional
import folium
from folium.plugins import HeatMap
from .geoclip import GeoCLIP

class GeoCLIPService:
    def __init__(self, model: Optional[GeoCLIP] = None, device: str = "cpu"):
        self.device = device
        self.model = model.to(self.device) if model else GeoCLIP(from_pretrained=True).to(self.device)

    def predict(self, img_input: Union[str, Image.Image, torch.Tensor], top_k: int = 5) -> List[Tuple[float, float, float]]: 
        """Predict GPS coordinates for a given image."""
        if isinstance(img_input, str):
            img = Image.open(img_input)
        elif isinstance(img_input, Image.Image):
            img = img_input
        elif isinstance(img_input, torch.Tensor):
            img = Image.fromarray(img_input.numpy())
        
        img = self.model.image_encoder.preprocess_image(img).to(self.device)
        gps_gallery = self.model.gps_gallery.to(self.device)
        
        logits = self.model.forward(img, gps_gallery)
        probs = logits.softmax(dim=-1).cpu()

        top_preds = torch.topk(probs, top_k, dim=1)
        coords = self.model.gps_gallery[top_preds.indices[0]]
        confs = top_preds.values[0]

        return [(coords[i, 0].item(), coords[i, 1].item(), confs[i].item()) for i in range(top_k)]

    def create_heatmap(self, predictions: List[Tuple[float, float, float]], center: Optional[Tuple[float, float]] = None) -> folium.Map:
        """Generate a heatmap from predictions."""
        coords, weights = zip(*[(pred[:2], pred[2]) for pred in predictions])
        norm_weights = [w / sum(weights) for w in weights]
        
        center = center or (sum(lat for lat, lon in coords) / len(coords), sum(lon for lat, lon in coords) / len(coords))
        m = folium.Map(location=center, zoom_start=2.2)
        
        HeatMap(list(zip(*coords, norm_weights)), gradient={0.0: '#932667', 1.0: '#fcfdbf'}).add_to(m)
        folium.Marker(location=coords[0], popup=f"Top Prediction: {coords[0]} with confidence {norm_weights[0]:.4f}", icon=folium.Icon(color='orange', icon='star')).add_to(m)

        return m
