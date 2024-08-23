import torch
import torch.nn as nn
from .rff import GaussianEncoding
from .misc import file_dir

# Constants for Equal Earth projection
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L):
    """
    Apply Equal Earth projection to latitude and longitude coordinates.

    Args:
        L (torch.Tensor): Input tensor of shape (N, 2) containing [latitude, longitude] pairs.

    Returns:
        torch.Tensor: Projected coordinates of shape (N, 2).
    """
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180

class LocationEncoderCapsule(nn.Module):
    """
    A capsule for encoding location data at a specific scale.

    This module applies Gaussian encoding followed by a series of linear layers.

    Attributes:
        km (float): The scale (in kilometers) at which this capsule operates.
        capsule (nn.Sequential): The main encoding network.
        head (nn.Sequential): The output layer of the capsule.
    """

    def __init__(self, sigma):
        """
        Initialize the LocationEncoderCapsule.

        Args:
            sigma (float): The scale parameter for Gaussian encoding.
        """
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        """
        Forward pass of the LocationEncoderCapsule.

        Args:
            x (torch.Tensor): Input tensor of projected coordinates.

        Returns:
            torch.Tensor: Encoded location features.
        """
        x = self.capsule(x)
        x = self.head(x)
        return x

class LocationEncoder(nn.Module):
    """
    Encodes geographical coordinates into feature vectors.

    This module uses multiple LocationEncoderCapsules at different scales
    to create a multi-scale representation of the location.

    Attributes:
        sigma (list): List of scale parameters for the capsules.
        n (int): Number of capsules.
    """

    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=True):
        """
        Initialize the LocationEncoder.

        Args:
            sigma (list): List of scale parameters for the capsules.
            from_pretrained (bool): Whether to load pre-trained weights.
        """
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        for i, s in enumerate(self.sigma):
            self.add_module(f'LocEnc{i}', LocationEncoderCapsule(sigma=s))
        if from_pretrained:
            self._load_weights()

    def _load_weights(self):
        """Load pre-trained weights for the LocationEncoder."""
        self.load_state_dict(torch.load(f"{file_dir}/weights/location_encoder_weights.pth"))

    def forward(self, location):
        """
        Forward pass of the LocationEncoder.

        Args:
            location (torch.Tensor): Input tensor of shape (N, 2) containing [latitude, longitude] pairs.

        Returns:
            torch.Tensor: Encoded location features of shape (N, 512).
        """
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], 512).to(location.device)
        for i in range(self.n):
            location_features += self._modules[f'LocEnc{i}'](location)
        
        return location_features
