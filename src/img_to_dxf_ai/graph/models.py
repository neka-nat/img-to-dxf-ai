import math

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: int = Field(...)
    x: float = Field(..., description="X position")
    y: float = Field(..., description="Y position")


Polyline = list[tuple[float, float]]


class Circle(BaseModel):
    center: tuple[float, float]
    radius: float
    start_angle: float
    end_angle: float

    def translate(self, dx: float, dy: float) -> "Circle":
        return Circle(
            center=(self.center[0] + dx, self.center[1] + dy),
            radius=self.radius,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
        )


LineType = Polyline | Circle


class Edge(BaseModel):
    id: int = Field(...)
    start_node: int = Field(..., description="Start node id")
    end_node: int = Field(..., description="End node id")
    edge_line: LineType = Field(...)


class Graph(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def get_node_dict(self) -> dict[int, Node]:
        return {node.id: node for node in self.nodes}

    def get_bbox(self) -> tuple[float, float, float, float]:
        min_x = min([n.x for n in self.nodes])
        min_y = min([n.y for n in self.nodes])
        max_x = max([n.x for n in self.nodes])
        max_y = max([n.y for n in self.nodes])
        return min_x, min_y, max_x, max_y

    def get_center(self) -> tuple[float, float]:
        bbox = self.get_bbox()
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def get_size(self) -> float:
        bbox = self.get_bbox()
        return max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    def translate(self, dx: float, dy: float) -> "Graph":
        nodes = [Node(id=n.id, x=n.x + dx, y=n.y + dy) for n in self.nodes]
        edges = [
            Edge(
                id=e.id,
                start_node=e.start_node,
                end_node=e.end_node,
                edge_line=(
                    e.edge_line.translate(dx, dy)
                    if isinstance(e.edge_line, Circle)
                    else [(x + dx, y + dy) for x, y in e.edge_line]
                ),
            )
            for e in self.edges
        ]
        return Graph(nodes=nodes, edges=edges)

    def add_circle(
        self,
        center: tuple[float, float],
        radius: float,
        start_angle: float = 0.0,
        end_angle: float = 2.0 * math.pi,
    ) -> "Graph":
        circle = Circle(center=center, radius=radius, start_angle=start_angle, end_angle=end_angle)
        self.nodes.append(Node(id=len(self.nodes), x=center[0], y=center[1]))
        self.edges.append(
            Edge(
                id=len(self.edges),
                start_node=len(self.nodes),
                end_node=len(self.nodes),
                edge_line=circle,
            )
        )
        return self

    @classmethod
    def merge(cls, *graphs: "Graph") -> "Graph":
        return Graph(
            nodes=sum([graph.nodes for graph in graphs], []),
            edges=sum([graph.edges for graph in graphs], []),
        )
