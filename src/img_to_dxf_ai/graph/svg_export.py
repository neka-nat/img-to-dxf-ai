from .models import Graph, Circle


def graph_to_svg(
    graph: Graph,
    width=500,
    height=500,
    stroke_width=1,
    stroke_color="#000",
    node_radius=1,
) -> str:
    """
    Export the graph to SVG.

    Args:
        graph (Graph): The graph.
        width (int): The width of the SVG.
        height (int): The height of the SVG.
        stroke_width (int): The width of the stroke.
        stroke_color (str): The color of the stroke.
        node_radius (int): The radius of the node.

    Returns:
        str: The SVG string.
    """
    # 座標範囲を求める
    xs = []
    ys = []
    for node in graph.nodes:
        xs.append(node.x)
        ys.append(node.y)
    for edge in graph.edges:
        if isinstance(edge.edge_line, list):
            for x, y in edge.edge_line:
                xs.append(x)
                ys.append(y)
        elif isinstance(edge.edge_line, Circle):
            # Circle の場合は円のバウディングボックスを求める
            (cx, cy), r = edge.edge_line.center, edge.edge_line.radius
            xs.append(cx - r)
            xs.append(cx + r)
            ys.append(cy - r)
            ys.append(cy + r)
        else:
            raise ValueError(f"Unknown edge type: {type(edge.edge_line)}")

    if len(xs) == 0 or len(ys) == 0:
        # 空の場合
        return "<svg></svg>"

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # SVG座標変換のためのスケール
    # データをwidth, height内に収まるようにする
    scale_x = (width - 10) / (max_x - min_x) if max_x != min_x else 1.0
    scale_y = (height - 10) / (max_y - min_y) if max_y != min_y else 1.0
    scale = min(scale_x, scale_y)

    def tx(x):
        return (x - min_x) * scale + 5

    def ty(y):
        return (y - min_y) * scale + 5

    svg = []
    svg.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )
    # エッジ描画
    for edge in graph.edges:
        if isinstance(edge.edge_line, list):
            points_str = " ".join([f"{tx(x)},{ty(y)}" for (x, y) in edge.edge_line])
            svg.append(
                f'<polyline points="{points_str}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}"/>'
            )
        elif isinstance(edge.edge_line, Circle):
            # Circle の場合は円を描画
            (cx, cy), r = edge.edge_line.center, edge.edge_line.radius
            svg.append(
                f'<circle cx="{tx(cx)}" cy="{ty(cy)}" r="{r * scale}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}"/>'
            )
        else:
            raise ValueError(f"Unknown edge type: {type(edge.edge_line)}")

    # ノード描画
    for node in graph.nodes:
        svg.append(
            f'<circle cx="{tx(node.x)}" cy="{ty(node.y)}" r="{node_radius}" fill="red"/>'
        )

    svg.append("</svg>")
    return "\n".join(svg)
