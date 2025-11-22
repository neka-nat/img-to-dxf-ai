import json
from datetime import datetime
from pathlib import Path

from PIL import Image
from loguru import logger
from tqdm import tqdm

from .colorize import colorize_img
from .graph.graph_pipeline import build_line_graph_from_image
from .graph.models import Graph
from .img_proc.circle_detection import detect_circles
from .detection import extract_segmentation_masks
from .separation import separate_images
from .utils import pil_to_cv2


def convert_to_outline_graph(image: Image.Image, resume_from_last_log: bool = False) -> str:
    if resume_from_last_log:
        log_dir = list(Path(".").glob("logs_*"))
        log_dir.sort(key=lambda x: x.name, reverse=True)
        log_dir = log_dir[0]
        colorized_images = [
            Image.open(log_dir / f"colorized_{i}.png")
            for i in range(len(list(log_dir.glob("colorized_*.png"))))
        ]
        with open(log_dir / "merged_boxes.json", "r") as f:
            merged_boxes = json.load(f)
    else:
        log_dir = Path(f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        log_dir.mkdir(parents=True, exist_ok=True)
        merged_boxes = extract_segmentation_masks(image)
        with open(log_dir / "merged_boxes.json", "w") as f:
            json.dump(merged_boxes, f)
        cropped_images = [image.crop(box) for box in merged_boxes]
        for i, cropped_image in enumerate(cropped_images):
            cropped_image.save(f"{log_dir}/cropped_{i}.png")
        logger.info(f"Found {len(cropped_images)} views")
        logger.info(f"Colorizing...")
        colorized_images = [colorize_img(image) for image in tqdm(cropped_images)]
        for i, colorized_image in enumerate(colorized_images):
            colorized_image.save(f"{log_dir}/colorized_{i}.png")
        logger.info(f"Building graphs...")
    graphs = []
    for i, (colorized_image, box) in enumerate(zip(colorized_images, merged_boxes)):
        separated_images = separate_images(colorized_image)
        for j, (name, image) in enumerate(separated_images.items()):
            image.save(f"{log_dir}/separated_{name}_{j}_{i}.png")
        graph = build_line_graph_from_image(pil_to_cv2(separated_images["outlines"]))
        circles = detect_circles(pil_to_cv2(separated_images["circles"]))
        for circle in circles:
            graph.add_circle(circle.center, circle.radius)
        graph = graph.translate(box[0], box[1])
        graphs.append(graph)
    return Graph.merge(*graphs)
