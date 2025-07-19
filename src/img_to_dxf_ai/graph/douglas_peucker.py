import math


def douglas_peucker(
    points: list[tuple[float, float]], epsilon: float
) -> list[tuple[float, float]]:
    """Douglas-Peucker algorithm

    Args:
        points: List of points
        epsilon: Tolerance for simplification

    Returns:
        Simplified points
    """

    # 再帰関数定義
    def dp_recursive(pts):
        if len(pts) < 3:
            return pts
        # 始点と終点
        start, end = pts[0], pts[-1]

        # 最大距離
        max_dist = 0.0
        index = 0

        # 始点end,終点startを直線として、他の点の距離を調べる
        for i in range(1, len(pts) - 1):
            d = perpendicular_distance(pts[i], start, end)
            if d > max_dist:
                index = i
                max_dist = d

        if max_dist > epsilon:
            # 分割
            left = dp_recursive(pts[: index + 1])
            right = dp_recursive(pts[index:])
            return left[:-1] + right
        else:
            return [start, end]

    return dp_recursive(points)


def perpendicular_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    # pointとstart-end線分との垂直距離を計算
    (x0, y0) = point
    (x1, y1) = start
    (x2, y2) = end
    if x1 == x2 and y1 == y2:
        return math.dist(point, start)
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return numerator / denominator
