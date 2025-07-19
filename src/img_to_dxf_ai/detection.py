import io
import base64
import json
import numpy as np

from google import genai
from google.genai import types
from PIL import Image
from loguru import logger
from pydantic import BaseModel, ConfigDict


class SegmentationMask(BaseModel):
    box_2d: list[int]
    mask: np.ndarray
    label: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


def parse_json(json_output: str) -> str:
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return output


def extract_segmentation_masks(image: Image.Image) -> list[SegmentationMask]:
    client = genai.Client()
    image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    labels = "front view, side view, top view and bottom view"

    prompt = f"""
    Give the segmentation masks for the {labels}.
    Output a JSON list of segmentation masks where each entry contains the 2D
    bounding box in the key "box_2d", the segmentation mask in key "mask", and
    the text label in the key "label". Use descriptive labels.
    """

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, image], # Pillow images can be directly passed as inputs (which will be converted by the SDK)
        config=config
    )

    # Parse JSON response
    logger.debug(f"Response: {response.text}")
    items = json.loads(parse_json(response.text))

    segmentation_masks = []
    # Process each mask
    for item in items:
        # Get bounding box coordinates
        box = item["box_2d"]
        y0 = int(box[0] / 1000 * image.size[1])
        x0 = int(box[1] / 1000 * image.size[0])
        y1 = int(box[2] / 1000 * image.size[1])
        x1 = int(box[3] / 1000 * image.size[0])

        # Skip invalid boxes
        if y0 >= y1 or x0 >= x1:
            continue

        # Process mask
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            continue

        # Remove prefix
        png_str = png_str.removeprefix("data:image/png;base64,")
        mask_data = base64.b64decode(png_str)
        mask = Image.open(io.BytesIO(mask_data))

        # Resize mask to match bounding box
        mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

        # Convert mask to numpy array for processing
        mask_array = np.array(mask)

        # Create segmentation mask
        segmentation_mask = SegmentationMask(
            box_2d=[x0, y0, x1, y1],
            mask=mask_array,
            label=item["label"]
        )
        segmentation_masks.append(segmentation_mask)

    return segmentation_masks


def _area(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = _intersection(a, b)
    if inter == 0:
        return 0.0
    return inter / (_area(a) + _area(b) - inter)


def _coverage(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """重なり面積 ÷ 小さい方の面積"""
    inter = _intersection(a, b)
    if inter == 0:
        return 0.0
    return inter / min(_area(a), _area(b))


def merge_boxes(
    segmentation_masks: list[SegmentationMask],
    iou_threshold: float = 0.5,
    coverage_threshold: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    boxes = [mask.box_2d for mask in segmentation_masks]
    merged: list[tuple[int, int, int, int]] = []

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        merged_here = False

        for i, m in enumerate(merged):
            if (
                _iou(box, m) >= iou_threshold
                and _coverage(box, m) >= coverage_threshold
            ):
                mx1, my1, mx2, my2 = m
                merged[i] = (
                    min(xmin, mx1),
                    min(ymin, my1),
                    max(xmax, mx2),
                    max(ymax, my2),
                )
                merged_here = True
                break

        if not merged_here:
            merged.append(box)

    # 連鎖的な重なりを取り切る
    if len(merged) < len(boxes):
        return merge_boxes(merged, iou_threshold, coverage_threshold)

    return merged


def add_margin(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    return xmin - margin, ymin - margin, xmax + margin, ymax + margin
