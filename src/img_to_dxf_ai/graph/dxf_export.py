import math
from typing import Iterable

from .models import Circle, Graph, Polyline


def graph_to_dxf(graph: Graph, layer: str = "0") -> str:
    """Convert a Graph into a minimal DXF string (R12-compatible)."""
    lines: list[str] = []

    y_values: list[float] = []
    for node in graph.nodes:
        y_values.append(node.y)
    for edge in graph.edges:
        if isinstance(edge.edge_line, list):
            y_values.extend(y for _, y in edge.edge_line)
        elif isinstance(edge.edge_line, Circle):
            center_y = edge.edge_line.center[1]
            radius = edge.edge_line.radius
            y_values.extend([center_y - radius, center_y + radius])

    min_y = min(y_values) if y_values else 0.0
    max_y = max(y_values) if y_values else 0.0

    def flip_y(y: float) -> float:
        # Mirror Y so DXF matches image coordinates (image Y grows downward).
        return min_y + max_y - y if y_values else y

    def mirror_arc_angles(start_deg: float, end_deg: float) -> tuple[float, float]:
        mirrored_start = (-end_deg) % 360.0
        mirrored_end = (-start_deg) % 360.0
        return mirrored_start, mirrored_end

    def add_code_pair(code: int | str, value: int | float | str) -> None:
        lines.append(str(code))
        lines.append(str(value))

    def add_polyline(points: Polyline) -> None:
        if not points:
            return
        closed = points[0] == points[-1]
        # POLYLINE header
        add_code_pair(0, "POLYLINE")
        add_code_pair(8, layer)
        add_code_pair(66, 1)  # Vertices follow
        add_code_pair(70, 1 if closed else 0)
        for x, y in points:
            add_code_pair(0, "VERTEX")
            add_code_pair(8, layer)
            add_code_pair(10, f"{x:.6f}")
            add_code_pair(20, f"{flip_y(y):.6f}")
            add_code_pair(30, "0.0")
        add_code_pair(0, "SEQEND")

    def add_circle(circle: Circle) -> None:
        start_deg = math.degrees(circle.start_angle)
        end_deg = math.degrees(circle.end_angle)
        is_full_circle = math.isclose(
            (circle.end_angle - circle.start_angle) % (2 * math.pi), 0.0, abs_tol=1e-3
        )
        if not is_full_circle:
            start_deg, end_deg = mirror_arc_angles(start_deg, end_deg)
        add_code_pair(0, "CIRCLE" if is_full_circle else "ARC")
        add_code_pair(8, layer)
        add_code_pair(10, f"{circle.center[0]:.6f}")
        add_code_pair(20, f"{flip_y(circle.center[1]):.6f}")
        add_code_pair(30, "0.0")
        add_code_pair(40, f"{circle.radius:.6f}")
        if not is_full_circle:
            add_code_pair(50, f"{start_deg:.6f}")
            add_code_pair(51, f"{end_deg:.6f}")

    def add_entities(edges: Iterable) -> None:
        for edge in edges:
            if isinstance(edge.edge_line, list):
                add_polyline(edge.edge_line)
            elif isinstance(edge.edge_line, Circle):
                add_circle(edge.edge_line)
            else:
                raise TypeError(f"Unsupported edge type: {type(edge.edge_line)}")

    # DXF sections
    add_code_pair(0, "SECTION")
    add_code_pair(2, "HEADER")
    add_code_pair(9, "$ACADVER")
    add_code_pair(1, "AC1015")
    add_code_pair(0, "ENDSEC")

    add_code_pair(0, "SECTION")
    add_code_pair(2, "ENTITIES")
    add_entities(graph.edges)
    add_code_pair(0, "ENDSEC")
    add_code_pair(0, "EOF")

    return "\n".join(lines)
