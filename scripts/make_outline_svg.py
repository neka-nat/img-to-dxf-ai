import argparse
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from img_to_dxf_ai.graph.svg_export import graph_to_svg
from img_to_dxf_ai.pipeline import convert_to_outline_graph

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=Path)
parser.add_argument("--resume-from-last-log", action="store_true")
args = parser.parse_args()

image = Image.open(args.image_path)
graph = convert_to_outline_graph(image, args.resume_from_last_log)
svg = graph_to_svg(graph)
with open(f"{args.image_path.with_suffix('.svg')}", "w") as f:
    f.write(svg)

