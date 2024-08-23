import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    """
    Image encoder using CLIP model for feature extraction.

    This class uses a pre-trained CLIP model to extract image features and
    then applies an MLP to further process these features.

    Attributes:
        CLIP (CLIPModel): Pre-trained CLIP model for feature extraction.
        image_processor (AutoProcessor): Processor for preparing images for CLIP.
        mlp (nn.Sequential): Multi-layer perceptron for feature processing.
    """

    def __init__(self):
        """Initialize the ImageEncoder with a pre-trained CLIP model and custom MLP."""
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512)
        )
        # Freeze CLIP parameters
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        """
        Preprocess the input image for CLIP model.

        Args:
            image: Input image (can be PIL Image, numpy array, or torch tensor).

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        """
        Forward pass of the ImageEncoder.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded image features.
        """
        x = self.CLIP.get_image_features(pixel_values=x)
        x = self.mlp(x)
        return x
