import cv2
import numpy as np
from PIL import Image

from .utils import pil_to_cv2, cv2_to_pil


lines_map = {
    "red": "dimensions",
    "black": "extensions",
    "blue": "outlines",
    "green": "circles"
}


def separate_images(image: Image.Image) -> dict[str, Image.Image]:
    cv2_image = pil_to_cv2(image)
    hsv  = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)

    ranges = {
        "red": [((  0,120,120), ( 10,255,255)),   # H=0±10°
                 ((170,120,120), (180,255,255))],  # H=180±10°（赤は 2 区間）
        "blue": [((100, 80, 50), (130,255,255))],  # H≈120°
        "green":[(( 40,100, 50), ( 80,255,255))],  # H≈60°
        "black":[((  0,  0,  0), (180,255, 60))],  # 低 V（暗い色）
    }

    masks  = {}
    for name, hsv_ranges in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in hsv_ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        masks[name] = mask

    images: dict[str, Image.Image] = {}
    for name, mask in masks.items():
        # 背景 = 白(255)、線 = 黒(0)
        bw = np.full_like(mask, 255)   # 全白
        bw[mask > 0] = 0               # 線ピクセルを黒に
        images[lines_map[name]] = cv2_to_pil(bw)

    return images
