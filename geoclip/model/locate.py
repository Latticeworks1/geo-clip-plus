"""
locate.py: Simplified interface for image geolocalization using GeoCLIP.

This module provides a user-friendly API for predicting geographical locations
from images using the GeoCLIP model.
"""

from typing import Union, List, Tuple, Optional
from PIL import Image
import torch

from .geoclip import GeoCLIP, GPSPrediction

def image(
    input: Union[str, Image.Image, torch.Tensor, List[Union[str, Image.Image, torch.Tensor]]],
    top_k: int = 1,
    model: Optional[GeoCLIP] = None,
    **kwargs
) -> List[List[Tuple[float, float, float]]]:
    """
    Predict geographical location(s) for the given image(s).

    This function provides a simple interface to the GeoCLIP model for image geolocalization.
    It can handle single images or batches of images, and returns predictions sorted by confidence.

    Args:
        input: A single image or a list of images. Each image can be:
               - A string containing the file path to the image
               - A PIL Image object
               - A PyTorch tensor of shape (C, H, W) or (B, C, H, W)
        top_k: The number of top predictions to return for each image. Defaults to 1.
        model: An optional pre-initialized GeoCLIP model. If not provided, a new model
               will be initialized with default settings.
        **kwargs: Additional keyword arguments to pass to GeoCLIP initialization if
                  a new model needs to be created.

    Returns:
        A list of lists containing tuples of (latitude, longitude, probability) for each input image.
        The outer list corresponds to input images, and the inner list contains up to `top_k` predictions
        for each image, sorted by descending probability.

    Examples:
        >>> from geoclip import locate
        >>> # Single image prediction
        >>> result = locate.image("path/to/image.jpg")
        >>> lat, lon, prob = result[0][0]
        >>> print(f"Predicted location: ({lat:.6f}, {lon:.6f}), Confidence: {prob:.4f}")

        >>> # Batch prediction with top 3 results
        >>> batch_results = locate.image(["image1.jpg", "image2.jpg"], top_k=3)
        >>> for i, predictions in enumerate(batch_results):
        ...     print(f"Image {i+1} predictions:")
        ...     for lat, lon, prob in predictions:
        ...         print(f"  ({lat:.6f}, {lon:.6f}) - Confidence: {prob:.4f}")
    """
    if model is None:
        model = GeoCLIP.from_pretrained(**kwargs)

    predictions = model.predict(input, top_k=top_k)
    return [[(pred.latitude, pred.longitude, pred.probability) for pred in img_preds] for img_preds in predictions]

# Alias for even shorter import
locate = image
