"""
Geoserve API Application

This module provides a FastAPI application for serving GeoCLIP model predictions.
It includes endpoints for image geolocation prediction and health checking.

The application supports automatic device selection (CUDA, MPS, or CPU) and
provides robust error handling and logging.

Attributes:
    logger (logging.Logger): Logger for the module.

Functions:
    create_app(): Creates and configures the FastAPI application.
    run_server(host: str, port: int): Runs the server with the specified host and port.

Usage:
    To run the server, execute this script directly or import and use the run_server function.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model.geoclip import GeoCLIP
from PIL import Image
from io import BytesIO
import logging
from typing import Dict, List, Tuple
import uvicorn
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger(__name__)

def select_device() -> torch.device:
    """
    Selects the most appropriate device for model execution.

    Returns:
        torch.device: The selected device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS is available. Using Metal GPU.")
        return torch.device("mps")
    else:
        logger.info("No GPU available. Using CPU.")
        return torch.device("cpu")

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.

    This function sets up CORS, initializes the GeoCLIP model,
    and defines the API endpoints.

    Returns:
        FastAPI: The configured FastAPI application.
    """
    app = FastAPI(
        title="Geoserve API",
        description="API for image geolocation prediction using GeoCLIP",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    device = select_device()
    model = GeoCLIP(from_pretrained=True).to(device)

    @app.post("/predict", response_model=Dict[str, List[Dict[str, float]]])
    async def predict(file: UploadFile = File(...), top_k: int = 5) -> Dict[str, List[Dict[str, float]]]:
        """
        Predicts geolocation for the uploaded image.

        Args:
            file (UploadFile): The uploaded image file.
            top_k (int): The number of top predictions to return. Defaults to 5.

        Returns:
            Dict[str, List[Dict[str, float]]]: A dictionary containing the top predictions.

        Raises:
            HTTPException: If the prediction fails.
        """
        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            
            top_pred_gps, top_pred_prob = model.predict(image, top_k)
            
            predictions = [
                {"lat": float(lat), "lon": float(lon), "probability": float(prob)}
                for (lat, lon), prob in zip(top_pred_gps, top_pred_prob)
            ]
            
            return {"predictions": predictions}
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """
        Checks the health status of the API.

        Returns:
            Dict[str, str]: A dictionary indicating the API status.
        """
        return {"status": "healthy"}

    return app

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Runs the FastAPI server.

    Args:
        host (str): The host to bind the server to. Defaults to "0.0.0.0".
        port (int): The port to bind the server to. Defaults to 8000.
    """
    app = create_app()
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
