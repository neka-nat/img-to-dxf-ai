import numpy as np
from .models import Graph, Node, Edge


def build_graph_from_skeleton(img: np.ndarray) -> Graph:
    """
    グラフ抽出処理。
    1. ノード(交点,端点)ピクセルを特定
    2. ノードから出ている2近傍経路をDFS/BFSでたどり、別ノードまでエッジを確定
    3. 訪れたピクセルは再利用せず重複エッジを回避
    """
    # 前景ピクセルの集合
    foreground_pixels = np.argwhere(img > 0)
    pixel_set = set(tuple(p) for p in foreground_pixels)

    # 8近傍定義
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_neighbors(r, c):
        return [
            (r + dr, c + dc) for dr, dc in neighbors if (r + dr, c + dc) in pixel_set
        ]

    # ノード判定
    # 隣接前景数 = 1: 端点ノード
    # 隣接前景数 >=3: 分岐ノード
    # 隣接前景数 = 2: 通過点
    node_id_map: dict[tuple[int, int], int] = {}
    nodes = []
    next_node_id = 0
    for r, c in pixel_set:
        nbrs = get_neighbors(r, c)
        if len(nbrs) == 1 or len(nbrs) >= 3:
            node_id_map[(r, c)] = next_node_id
            nodes.append(Node(id=next_node_id, x=float(c), y=float(r)))
            next_node_id += 1

    visited = set()
    edges: list[Edge] = []
    next_edge_id = 0

    # エッジを探索する関数
    def trace_edge(start_node_pixel):
        """
        start_node_pixelから、2近傍ピクセルを辿って次のノードに到達するまでのルートを探索する。
        BFSまたはDFSで実装。
        """
        stack = []
        # start_node_pixelはノードなので、そこから出るnon-node方向のピクセルを探索
        start_nid = node_id_map[start_node_pixel]
        start_nbrs = get_neighbors(*start_node_pixel)
        # ノードから出る候補は通過点(2近傍)ピクセルのみ追跡
        for n in start_nbrs:
            if n not in node_id_map:  # 通過点への入口
                # エッジ探索スタックにプッシュ
                stack.append(
                    [start_node_pixel, n]
                )  # [ノードピクセル, 初回通過点ピクセル]

        # stackには複数のエッジ候補が入る場合がある(分岐ノードなら)
        found_edges = []
        while stack:
            path = stack.pop()  # pathはピクセル列
            current = path[-1]
            if current in visited:
                # 既に訪問済みのピクセルならこれ以上伸ばせない
                continue

            visited.add(current)
            nbrs = get_neighbors(*current)
            # currentがノードか？
            if current in node_id_map and current != start_node_pixel:
                # ゴールノード到達
                end_nid = node_id_map[current]
                # pathは start_node_pixel(ノード), ... , current(ノード) を含むエッジ
                # 途中は全部通過点
                polyline = [(float(p[1]), float(p[0])) for p in path]
                found_edges.append((start_nid, end_nid, polyline))
                continue
            else:
                # 通過点の場合
                # 通過点は2近傍を持つはず(ただしノイズでそうでない場合もある)
                # 次に進むべき通過点ピクセルを探す
                # 近傍から既存pathに入っていない、visitedでない、nodeでないピクセルを探す
                valid_next = []
                for pix in nbrs:
                    # path[-2]が直前のピクセルで戻らないように
                    if (
                        pix not in visited
                        and pix not in node_id_map
                        and pix not in path
                    ):
                        # 通過点ピクセルなら進めるか確認
                        # 通過点は2近傍しか持たないはず
                        # 最小限条件チェック
                        if len(get_neighbors(*pix)) in [1, 2]:
                            valid_next.append(pix)
                    elif pix in node_id_map and pix != start_node_pixel:
                        # すぐ隣がノードなら到達エッジ確定
                        new_path = path + [pix]
                        polyline = [(float(p[1]), float(p[0])) for p in new_path]
                        found_edges.append((start_nid, node_id_map[pix], polyline))

                # valid_nextが0の場合、死端（次ノードに行けない）
                # valid_nextが1の場合、その方向に進む
                # valid_nextが2以上の場合、分岐しているので、それぞれpush
                for nxt in valid_next:
                    new_path = path + [nxt]
                    stack.append(new_path)

        return found_edges

    # 全ノードからエッジ探索
    # ノード周囲の通過点を起点に、他ノードまでたどり着くルートを見つける
    # found_edgesは重複する可能性があるので、重複排除
    seen_edges = set()
    for node_pixel, nid in node_id_map.items():
        edges_from_node = trace_edge(node_pixel)
        for snid, enid, polyline in edges_from_node:
            if snid > enid:
                # 順序を正規化
                snid, enid = enid, snid
                polyline = polyline[::-1]
            if (snid, enid) not in seen_edges:
                seen_edges.add((snid, enid))
                edges.append(
                    Edge(
                        id=next_edge_id,
                        start_node=snid,
                        end_node=enid,
                        edge_line=polyline,
                    )
                )
                next_edge_id += 1

    # 短すぎるエッジをフィルタリングするなど
    min_length = 2
    edges = [e for e in edges if len(e.edge_line) >= min_length]

    return Graph(nodes=nodes, edges=edges)
