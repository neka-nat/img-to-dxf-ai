import cv2
import numpy as np
from skimage.morphology import skeletonize

from .graph_builder import build_graph_from_skeleton
from .douglas_peucker import douglas_peucker
from .models import Graph


def skeletonize_drawing(color_image: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Skeletonize the drawing.

    Args:
        color_image (np.ndarray): The color image.

    Returns:
        np.ndarray: The skeletonized image.
    """
    if color_image.ndim == 3:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    elif color_image.ndim == 2:
        gray = color_image
    else:
        raise ValueError(f"Unsupported image shape: {color_image.shape}")

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
    )

    binary = cv2.bitwise_not(binary)

    opened_bool = binary > 0
    skeleton = skeletonize(opened_bool)

    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    if invert:
        skeleton_uint8 = cv2.bitwise_not(skeleton_uint8)
    return skeleton_uint8


def simplify_graph(graph: Graph, epsilon: float = 2.0) -> Graph:
    """
    Simplify the graph.

    Args:
        graph (Graph): The graph.
        epsilon (float): The tolerance for simplification.

    Returns:
        Graph: The simplified graph.
    """
    new_edges = []
    for e in graph.edges:
        simplified_poly = douglas_peucker(e.edge_line, epsilon)
        new_edges.append(e.copy(update={"edge_line": simplified_poly}))

    # ノードの再計算はここでは行わないが、エッジが簡略化されても
    # 開始・終了ノードは不変とする（必要なら再構築）
    # 簡略後、エッジの端点は元々のノードと同じ座標とは限らないが、
    # ここでは単純化のためそのままにしている。
    # 必要であれば、簡略後のエッジ末端に最も近いノードを再マッチングするなどの処理が必要。
    return Graph(nodes=graph.nodes, edges=new_edges)


def build_line_graph_from_image(img: np.ndarray, epsilon: float = 2.0) -> Graph:
    skeleton = skeletonize_drawing(img)
    graph = build_graph_from_skeleton(skeleton)
    simplified_graph = simplify_graph(graph, epsilon)
    return simplified_graph
