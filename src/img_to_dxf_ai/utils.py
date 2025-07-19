import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    if image.mode in ("RGB", "RGBA"):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif image.mode in ("L", "LA"):
        return np.array(image)
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:  # gray
        return Image.fromarray(image)
    elif image.ndim == 3:  # rgb
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
