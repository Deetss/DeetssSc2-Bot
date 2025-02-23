import cv2
import numpy as np

def resize_image(image, size=(84, 84)):
    """Resize image to given dimensions."""
    return cv2.resize(image, size)

def normalize_image(image):
    """Normalize pixel values to the range [0, 1]."""
    return image / 255.0

def preprocess_observation(obs):
    """
    Preprocess a raw observation (assumed to be a numpy array).
    You can extend this to perform additional operations such as 
    cropping, converting to grayscale, etc.
    """
    resized = resize_image(obs)
    normalized = normalize_image(resized)
    return normalized