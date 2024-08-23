"""
utils.py: Utility functions for the GeoCLIP project.

This module contains various helper functions used across the GeoCLIP project,
including data loading, seed setting, and geographical calculations.
"""

import random
import torch
import numpy as np
from typing import List, Tuple
import csv
import math

def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators in Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_gps_data(file_path: str) -> torch.Tensor:
    """
    Load GPS coordinates from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing GPS coordinates.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) containing the loaded GPS coordinates,
                      where N is the number of coordinates and each row is [latitude, longitude].

    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If the CSV file is empty or has an incorrect format.
    """
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            coordinates = [[float(row[0]), float(row[1])] for row in reader]
        
        if not coordinates:
            raise ValueError("The CSV file is empty.")
        
        return torch.tensor(coordinates)
    except FileNotFoundError:
        raise FileNotFoundError(f"GPS data file not found: {file_path}")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")

def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance between two points on the Earth's surface.

    This function uses the Haversine formula to compute the distance between two
    geographical coordinates given as (latitude, longitude) pairs.

    Args:
        coord1 (Tuple[float, float]): First coordinate (latitude, longitude) in degrees.
        coord2 (Tuple[float, float]): Second coordinate (latitude, longitude) in degrees.

    Returns:
        float: The distance between the two points in kilometers.
    """
    R = 6371  # Earth's radius in kilometers

    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def evaluate_predictions(true_coords: List[Tuple[float, float]], 
                         pred_coords: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """
    Evaluate the accuracy of geographical predictions.

    This function calculates various metrics to assess the quality of geographical predictions,
    including mean error distance, median error distance, and percentage of predictions within 1km.

    Args:
        true_coords (List[Tuple[float, float]]): List of true (latitude, longitude) coordinates.
        pred_coords (List[Tuple[float, float]]): List of predicted (latitude, longitude) coordinates.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - Mean error distance in kilometers
            - Median error distance in kilometers
            - Percentage of predictions within 1km of the true location

    Raises:
        ValueError: If the input lists have different lengths.
    """
    if len(true_coords) != len(pred_coords):
        raise ValueError("The number of true coordinates must match the number of predictions.")

    distances = [haversine_distance(true, pred) for true, pred in zip(true_coords, pred_coords)]
    
    mean_distance = sum(distances) / len(distances)
    median_distance = sorted(distances)[len(distances) // 2]
    within_1km = sum(1 for d in distances if d <= 1) / len(distances) * 100

    return mean_distance, median_distance, within_1km
