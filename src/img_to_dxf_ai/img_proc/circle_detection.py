import math

import cv2
import numpy as np
from pydantic import BaseModel


class Circle(BaseModel):
    center: tuple[float, float]
    radius: float


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """前処理：グレースケール化、ノイズ低減、コントラスト強調など"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return gray


def edge_detection(img: np.ndarray) -> np.ndarray:
    """Cannyエッジ検出を行う"""
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    return edges


def ransac_circle_fitting(
    edge_points: list,
    max_iter: int = 2000,
    dist_threshold: float = 2.0,
    min_inliers: int = 200,
    max_radius: float = 500.0,
) -> tuple[tuple[float, float, float] | None, np.ndarray | None]:
    if len(edge_points) < 3:
        return None, None

    best_circle = None
    best_inliers = 0
    pts = np.array(edge_points)
    best_inliers_mask = np.zeros(len(pts), dtype=bool)

    for _ in range(max_iter):
        # 3点ランダムサンプリング
        sample = pts[np.random.choice(len(pts), 3, replace=False)]
        c = fit_circle_3pts(sample[0], sample[1], sample[2])
        if c is None:
            continue
        cx, cy, r = c
        if r > max_radius:
            continue

        if r < 3.0:
            continue

        # インライア評価
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        inliers_mask = np.abs(dists - r) < dist_threshold
        inliers = np.sum(inliers_mask)
        if inliers > best_inliers and inliers >= min_inliers:
            best_inliers = inliers
            best_inliers_mask = inliers_mask
            best_circle = (cx, cy, r)

    if best_circle is None:
        return None, None
    return best_circle, best_inliers_mask


def fit_circle_3pts(p1: tuple, p2: tuple, p3: tuple) -> tuple:
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    # 行列式Dの計算(3点を通る円の存在条件)
    # D = 2 * det|x1 y1 1
    #             x2 y2 1
    #             x3 y3 1|
    A = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    D = 2.0 * np.linalg.det(A)

    if abs(D) < 1e-12:
        # 3点が直線上等で円が定まらない場合
        return None

    # 各点のx^2+y^2を計算
    x1s = x1**2 + y1**2
    x2s = x2**2 + y2**2
    x3s = x3**2 + y3**2

    # 中心 (cx, cy) を求める公式:
    # cx = [ (x1²+y1²)(y2 - y3) + (x2²+y2²)(y3 - y1) + (x3²+y3²)(y1 - y2) ] / D
    # cy = [ (x1²+y1²)(x3 - x2) + (x2²+y2²)(x1 - x3) + (x3²+y3²)(x2 - x1) ] / D
    cx = (x1s * (y2 - y3) + x2s * (y3 - y1) + x3s * (y1 - y2)) / D
    cy = (x1s * (x3 - x2) + x2s * (x1 - x3) + x3s * (x2 - x1)) / D

    # 半径rは任意の点との距離で計算可能
    r = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)

    return (cx, cy, r)


def extract_edge_points(edges: np.ndarray) -> np.ndarray:
    """エッジ画像からエッジ点群を抽出"""
    points = np.column_stack(np.where(edges > 0))
    # (y,x) -> (x,y)
    points = points[:, [1, 0]]
    return points


def find_similar_circle(circles: list[Circle], circle: Circle) -> bool:
    cx, cy, r = circle
    for c in circles:
        ccx, ccy, cr = c
        if (
            np.linalg.norm(np.array((ccx, ccy)) - np.array((cx, cy))) < r * 0.01 and
            abs(cr - r) < r * 0.01
        ):
            return True
    return False


def detect_circles(img: np.ndarray) -> list[Circle]:
    gray = preprocess_image(img)
    edges = edge_detection(gray)
    edge_points = extract_edge_points(edges)
    circles = []
    while True:
        circle, inliers_mask = ransac_circle_fitting(edge_points)
        if circle is None:
            break
        if find_similar_circle(circles, circle):
            continue
        circles.append(circle)
        edge_points = edge_points[~inliers_mask]
    return [Circle(center=circle[:2], radius=circle[2]) for circle in circles]
