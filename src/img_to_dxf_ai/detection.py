from PIL import Image

from vlm_recog.detection import detect
from vlm_recog.bbox_utils import merge_boxes, add_margin


def extract_segmentation_masks(image: Image.Image) -> list[tuple[int, int, int, int]]:
    results = detect(image, ["正面図", "背面図", "側面図"])
    merged_boxes = merge_boxes([box.box_2d for box in results], iou_threshold=0.5, coverage_threshold=0.5)
    merged_boxes = [add_margin(box, ((box[2] - box[0]) + (box[3] - box[1])) / 2.0 * 0.03) for box in merged_boxes]
    return merged_boxes
