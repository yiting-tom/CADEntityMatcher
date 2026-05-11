"""
cad_package_parser.py
=====================
自動從 DXF 圖面中偵測半導體封裝結構：
  - Substrate / Lid_outer / Inner_lid
  - Die area (BGA balls) + Cavity
  - SMD groups（body + marks，各種組合）
  - Fiducial marks（四角）
  - Pin-1 marker（front view 右上角三角形）

依賴：ezdxf, numpy, pandas, scipy, scikit-learn
安裝：pip install ezdxf numpy pandas scipy scikit-learn
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import ezdxf
import numpy as np
import pandas as pd
from scipy.ndimage import label as scipy_label
from sklearn.cluster import DBSCAN


# ─────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────


@dataclass
class Bbox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def w(self) -> float:
        return self.xmax - self.xmin

    @property
    def h(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return (
            max(self.w, self.h) / min(self.w, self.h)
            if min(self.w, self.h) > 0
            else 999
        )

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def contains_bbox(self, other: "Bbox") -> bool:
        return (
            self.xmin < other.xmin
            and self.ymin < other.ymin
            and self.xmax > other.xmax
            and self.ymax > other.ymax
        )

    def corners(self):
        return [
            (self.xmin, self.ymin),
            (self.xmin, self.ymax),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
        ]

    def __repr__(self):
        return f"Bbox(x={self.xmin:.1f}~{self.xmax:.1f}, y={self.ymin:.1f}~{self.ymax:.1f})"


@dataclass
class RectEntity:
    """包裝 LWPOLYLINE 矩形的資料"""

    entity: object  # ezdxf entity
    bbox: Bbox
    layer: str
    n_pts: int  # 頂點數（3=三角, 4=矩形, ...）

    @property
    def area(self) -> float:
        return self.bbox.area

    @property
    def cx(self) -> float:
        return self.bbox.cx

    @property
    def cy(self) -> float:
        return self.bbox.cy


@dataclass
class SmdGroup:
    """一個 SMD 的完整資訊"""

    body: Optional[RectEntity]  # 可能沒有
    marks: list[RectEntity]  # 0, 1, 或 2 個
    group_type: str  # complete / marks_only / body_only / body_one_mark
    side: str = ""  # top / bottom / left / right（分配後填入）
    is_inner: bool = False  # True = cavity 內部的 SMD

    @property
    def center(self) -> tuple[float, float]:
        if self.body:
            return self.body.cx, self.body.cy
        xs = [m.cx for m in self.marks]
        ys = [m.cy for m in self.marks]
        return float(np.mean(xs)), float(np.mean(ys))

    @property
    def bbox(self) -> Bbox:
        members = ([self.body] if self.body else []) + self.marks
        return Bbox(
            xmin=min(m.bbox.xmin for m in members),
            ymin=min(m.bbox.ymin for m in members),
            xmax=max(m.bbox.xmax for m in members),
            ymax=max(m.bbox.ymax for m in members),
        )

    def __repr__(self):
        cx, cy = self.center
        return (
            f"SmdGroup(type={self.group_type}, side={self.side}, "
            f"inner={self.is_inner}, center=({cx:.1f},{cy:.1f}))"
        )


@dataclass
class PackageResult:
    """一個封裝圖實例的完整解析結果"""

    substrate: Bbox
    substrate_entities: list  # 組成 substrate 的原始 entities
    lid_outer: Optional[Bbox]
    lid_outer_entities: list  # 組成 lid_outer 的原始 entities
    inner_lid: Optional[Bbox]
    inner_lid_entities: list  # 組成 inner_lid 的原始 entities
    die_bbox: Bbox
    bga_entities: list  # 所有 BGA ball entities
    cavity: Optional[Bbox]
    bga_count: int
    is_front_view: bool  # 有 Pin-1 → front view
    pin1: Optional[RectEntity]
    fiducial_groups: list  # [{"corner_idx", "shape", "entities"}]
    smd_groups: list[SmdGroup]  # 所有 SMD（含 inner）

    def summary(self) -> str:
        lines = ["=" * 50]
        lines.append(f"Substrate  : {self.substrate}")
        lines.append(f"Lid_outer  : {self.lid_outer or '無'}")
        lines.append(f"Inner_lid  : {self.inner_lid or '無'}")
        lines.append(f"Die area   : {self.die_bbox}")
        lines.append(f"Cavity     : {self.cavity or '無'}")
        lines.append(f"BGA balls  : {self.bga_count}")
        lines.append(f"Front view : {self.is_front_view}")
        outer = [s for s in self.smd_groups if not s.is_inner]
        inner = [s for s in self.smd_groups if s.is_inner]
        lines.append(f"SMD outer  : {len(outer)} 個")
        for side in ("top", "bottom", "left", "right"):
            n = sum(1 for s in outer if s.side == side)
            if n:
                lines.append(f"  {side:6s}: {n}")
        if inner:
            lines.append(f"SMD inner  : {len(inner)} 個")
        lines.append("=" * 50)
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 幾何工具
# ─────────────────────────────────────────────


def _entity_bbox(entity) -> Optional[Bbox]:
    """嘗試從任意 entity 取得 bbox"""
    t = entity.dxftype()
    try:
        if t == "LWPOLYLINE":
            pts = list(entity.get_points())
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return Bbox(min(xs), min(ys), max(xs), max(ys))
        elif t == "CIRCLE":
            r = entity.dxf.radius
            c = entity.dxf.center
            return Bbox(c.x - r, c.y - r, c.x + r, c.y + r)
        elif t == "LINE":
            s, e = entity.dxf.start, entity.dxf.end
            return Bbox(min(s.x, e.x), min(s.y, e.y), max(s.x, e.x), max(s.y, e.y))
        elif t == "INSERT":
            p = entity.dxf.insert
            return Bbox(p.x, p.y, p.x, p.y)
    except Exception:
        pass
    return None


def _entity_center(entity) -> tuple[Optional[float], Optional[float]]:
    bbox = _entity_bbox(entity)
    if bbox:
        return bbox.cx, bbox.cy
    return None, None


def _corner_distance(cx: float, cy: float, ref: Bbox) -> tuple[float, int]:
    """回傳到最近角點的距離與角點 index (0=左下,1=左上,2=右下,3=右上)"""
    corners = ref.corners()
    dists = [math.hypot(cx - fx, cy - fy) for fx, fy in corners]
    idx = int(np.argmin(dists))
    return dists[idx], idx


# ─────────────────────────────────────────────
# Step 1：收集所有閉合矩形
# ─────────────────────────────────────────────


def _collect_rects(msp) -> list[RectEntity]:
    return _collect_rects_from(list(msp))


def _collect_rects_from(entities: list) -> list[RectEntity]:
    rects = []
    for e in entities:
        if e.dxftype() != "LWPOLYLINE":
            continue
        if not e.is_closed:
            continue
        pts = list(e.get_points())
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w < 0.1 or h < 0.1:
            continue
        rects.append(
            RectEntity(
                entity=e,
                bbox=Bbox(min(xs), min(ys), max(xs), max(ys)),
                layer=e.dxf.layer,
                n_pts=len(pts),
            )
        )
    return sorted(rects, key=lambda r: r.area, reverse=True)


def _collect_circles(msp) -> pd.DataFrame:
    return _collect_circles_from(list(msp))


def _collect_circles_from(entities: list) -> pd.DataFrame:
    rows = []
    for e in entities:
        if e.dxftype() != "CIRCLE":
            continue
        rows.append(
            {
                "cx": e.dxf.center.x,
                "cy": e.dxf.center.y,
                "r": e.dxf.radius,
                "entity": e,
            }
        )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["cx", "cy", "r", "entity"])
    )


# ─────────────────────────────────────────────
# Step 3：找封裝圖 substrate 實例
# ─────────────────────────────────────────────


def _find_substrates(
    rects: list[RectEntity],
    df_circles: pd.DataFrame,
    min_bga: int = 10,
    max_aspect: float = 1.8,
) -> list[RectEntity]:
    """
    封裝圖 substrate 條件：
      1. 長寬比 < max_aspect（近似正方形）
      2. 內部 CIRCLE 數量 >= min_bga
    只保留「最外層」的候選（不被其他候選包含的）
    """
    candidates = []
    for r in rects:
        if r.bbox.aspect > max_aspect:
            continue
        if df_circles.empty:
            continue
        mask = (
            (df_circles["cx"] >= r.bbox.xmin)
            & (df_circles["cx"] <= r.bbox.xmax)
            & (df_circles["cy"] >= r.bbox.ymin)
            & (df_circles["cy"] <= r.bbox.ymax)
        )
        if mask.sum() >= min_bga:
            candidates.append(r)

    # 只保留「最外層」：不被任何其他候選包含
    final = []
    for r in candidates:
        is_contained = any(
            (
                other.bbox.xmin <= r.bbox.xmin
                and other.bbox.ymin <= r.bbox.ymin
                and other.bbox.xmax >= r.bbox.xmax
                and other.bbox.ymax >= r.bbox.ymax
                and other is not r
            )
            for other in candidates
        )
        if not is_contained:
            final.append(r)
    return final


# ─────────────────────────────────────────────
# Step 4：找 fiducial marks
# ─────────────────────────────────────────────


def _find_fiducials(
    all_entities, ref: Bbox, threshold_ratio: float = 0.12
) -> dict[int, list]:
    """
    找四角附近的 entity（不管形狀）
    回傳 {corner_idx: [entity, ...]}
    """
    threshold = min(ref.w, ref.h) * threshold_ratio
    fiducials: dict[int, list] = {0: [], 1: [], 2: [], 3: []}

    for e in all_entities:
        t = e.dxftype()
        if t in ("TEXT", "MTEXT", "DIMENSION", "ATTDEF", "ATTRIB"):
            continue
        cx, cy = _entity_center(e)
        if cx is None:
            continue
        dist, idx = _corner_distance(cx, cy, ref)
        if dist < threshold:
            fiducials[idx].append(e)

    return fiducials


# ─────────────────────────────────────────────
# Step 5：找 Pin-1（front view 右上角三角形）
# ─────────────────────────────────────────────


def _find_pin1(
    rects: list[RectEntity], substrate: Bbox, threshold_ratio: float = 0.15
) -> Optional[RectEntity]:
    """
    Pin-1 = 在 substrate 右上角附近的三角形（3頂點 LWPOLYLINE）
    """
    threshold = min(substrate.w, substrate.h) * threshold_ratio
    best = None
    best_dist = float("inf")

    for r in rects:
        if r.n_pts != 3:
            continue
        # 在右上角附近
        dist = math.hypot(r.cx - substrate.xmax, r.cy - substrate.ymax)
        if dist < threshold and dist < best_dist:
            best_dist = dist
            best = r

    return best


# ─────────────────────────────────────────────
# Step 6：找 BGA balls → die_bbox + cavity
# ─────────────────────────────────────────────


def _find_bga_and_die(
    df_circles: pd.DataFrame, search_bbox: Bbox
) -> tuple[pd.DataFrame, Bbox]:
    """在 search_bbox 內找 BGA balls，回傳 bga_df 與 die_bbox"""
    mask = (
        (df_circles["cx"] >= search_bbox.xmin)
        & (df_circles["cx"] <= search_bbox.xmax)
        & (df_circles["cy"] >= search_bbox.ymin)
        & (df_circles["cy"] <= search_bbox.ymax)
    )
    local = df_circles[mask]
    if local.empty:
        return pd.DataFrame(), search_bbox

    # radius 最多的 = BGA ball
    bga_radius = local["r"].value_counts().index[0]
    bga_df = local[local["r"] == bga_radius]

    die_bbox = Bbox(
        xmin=float(bga_df["cx"].min()),
        ymin=float(bga_df["cy"].min()),
        xmax=float(bga_df["cx"].max()),
        ymax=float(bga_df["cy"].max()),
    )
    return bga_df, die_bbox


def _detect_cavity(bga_df: pd.DataFrame, grid_size: float = None) -> Optional[Bbox]:
    """
    用密度網格找 BGA 中央的空洞（cavity）
    grid_size 預設為 BGA 間距的估算值
    """
    if bga_df.empty or len(bga_df) < 9:
        return None

    xs = bga_df["cx"].values
    ys = bga_df["cy"].values

    if grid_size is None:
        # 估算 BGA 間距
        sorted_x = np.sort(np.unique(np.round(xs, 0)))
        if len(sorted_x) > 1:
            grid_size = float(np.median(np.diff(sorted_x))) * 1.2
        else:
            grid_size = (xs.max() - xs.min()) / 10

    x_bins = np.arange(xs.min() - grid_size, xs.max() + grid_size * 2, grid_size)
    y_bins = np.arange(ys.min() - grid_size, ys.max() + grid_size * 2, grid_size)

    if len(x_bins) < 3 or len(y_bins) < 3:
        return None

    grid, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])

    # 外圍補 1（確保空洞在內部）
    padded = np.pad(grid, 1, constant_values=1)
    empty = (padded == 0).astype(int)

    labeled, n_features = scipy_label(empty)
    if n_features == 0:
        return None

    # 找最大的連通空白區，且必須不碰到 padded 邊界
    best_label = None
    best_size = 0
    for lbl in range(1, n_features + 1):
        region = labeled == lbl
        # 排除碰到邊界的區域
        if (
            region[0, :].any()
            or region[-1, :].any()
            or region[:, 0].any()
            or region[:, -1].any()
        ):
            continue
        size = region.sum()
        if size > best_size:
            best_size = size
            best_label = lbl

    if best_label is None:
        return None

    # threshold：空洞至少要佔 BGA 有效格子數的 5%，且至少 6 個 cell
    bga_cells = int((grid > 0).sum())
    if bga_cells == 0:
        return None
    if best_size < 6:
        return None
    if best_size / bga_cells < 0.05:
        return None

    cells = np.where(labeled == best_label)
    # 去掉 padding 偏移
    xi = cells[0] - 1
    yi = cells[1] - 1
    xi = np.clip(xi, 0, len(x_bins) - 2)
    yi = np.clip(yi, 0, len(y_bins) - 2)

    return Bbox(
        xmin=float(x_bins[xi.min()]),
        ymin=float(y_bins[yi.min()]),
        xmax=float(x_bins[xi.max() + 1]),
        ymax=float(y_bins[yi.max() + 1]),
    )


# ─────────────────────────────────────────────
# Step 7：找各層 lid
# ─────────────────────────────────────────────


def _find_lid_layers(
    rects: list[RectEntity], substrate: Bbox, die_bbox: Bbox
) -> tuple[Optional[Bbox], list, Optional[Bbox], list]:
    """
    找包圍 die_bbox 的矩形，按面積由小到大排列。
    回傳 (lid_outer_bbox, lid_outer_entities,
           inner_lid_bbox, inner_lid_entities)
    """
    die_cx, die_cy = die_bbox.cx, die_bbox.cy

    surrounding = []
    for r in rects:
        b = r.bbox
        if not b.contains_point(die_cx, die_cy):
            continue
        if b.area <= die_bbox.area:
            continue
        if b.area >= substrate.area * 0.95:
            continue
        if not (
            substrate.xmin <= b.xmin
            and substrate.ymin <= b.ymin
            and substrate.xmax >= b.xmax
            and substrate.ymax >= b.ymax
        ):
            continue
        surrounding.append(r)

    surrounding = sorted(surrounding, key=lambda r: r.area)

    if len(surrounding) == 0:
        return None, [], None, []
    elif len(surrounding) == 1:
        r = surrounding[0]
        ratio = r.area / die_bbox.area
        if ratio < 1.8:
            return None, [], r.bbox, [r.entity]  # inner_lid
        else:
            return r.bbox, [r.entity], None, []  # lid_outer
    else:
        # 最小的 = inner_lid，最大的 = lid_outer
        return (
            surrounding[-1].bbox,
            [surrounding[-1].entity],
            surrounding[0].bbox,
            [surrounding[0].entity],
        )


# ─────────────────────────────────────────────
# Step 8：收集 SMD candidates
# ─────────────────────────────────────────────


def _collect_smd_candidates(
    rects: list[RectEntity],
    outer_boundary: Bbox,
    exclusion_zone: Bbox,
    fiducial_ids: set,
    pin1_id: Optional[int],
    substrate: Bbox,
    max_area_ratio: float = 0.05,
) -> list[RectEntity]:
    """
    收集在 outer_boundary 內、exclusion_zone 外的矩形
    排除 fiducial、Pin-1、substrate、lid 等大矩形
    max_area_ratio: SMD 面積不超過 outer_boundary 的這個比例
    """
    max_smd_area = outer_boundary.area * max_area_ratio
    candidates = []
    for r in rects:
        # 排除太大的矩形（substrate、lid 等）
        if r.area >= max_smd_area:
            continue
        # 排除 fiducial 和 Pin-1
        if id(r.entity) in fiducial_ids:
            continue
        if pin1_id and id(r.entity) == pin1_id:
            continue
        cx, cy = r.cx, r.cy
        if not outer_boundary.contains_point(cx, cy):
            continue
        if exclusion_zone is not None and exclusion_zone.contains_point(cx, cy):
            continue
        candidates.append(r)
    return candidates


# ─────────────────────────────────────────────
# Step 9：DBSCAN 分群 + SMD 角色分類
# ─────────────────────────────────────────────


def _auto_area_threshold(candidates: list[RectEntity]) -> float:
    """從面積分布自動找 body/mark 的分界點"""
    if not candidates:
        return 0.0
    areas = sorted([r.area for r in candidates], reverse=True)
    if len(areas) == 1:
        return areas[0] / 2
    gaps = [(areas[i] - areas[i + 1], i) for i in range(len(areas) - 1)]
    best_idx = max(gaps, key=lambda x: x[0])[1]
    return (areas[best_idx] + areas[best_idx + 1]) / 2


def _classify_group(members: list[RectEntity], area_threshold: float) -> SmdGroup:
    members_sorted = sorted(members, key=lambda r: r.area, reverse=True)
    n = len(members_sorted)
    areas = [r.area for r in members_sorted]

    if n == 1:
        r = members_sorted[0]
        if r.area >= area_threshold:
            return SmdGroup(body=r, marks=[], group_type="body_only")
        else:
            return SmdGroup(body=None, marks=[r], group_type="mark_only")

    elif n == 2:
        ratio = areas[0] / areas[1] if areas[1] > 0 else 999
        if ratio > 2.5:
            return SmdGroup(
                body=members_sorted[0],
                marks=[members_sorted[1]],
                group_type="body_one_mark",
            )
        else:
            return SmdGroup(body=None, marks=members_sorted, group_type="marks_only")

    elif n == 3:
        ratio = areas[0] / areas[1] if areas[1] > 0 else 999
        if ratio > 2.5:
            return SmdGroup(
                body=members_sorted[0], marks=members_sorted[1:], group_type="complete"
            )
        else:
            return SmdGroup(
                body=members_sorted[0], marks=members_sorted[1:], group_type="ambiguous"
            )

    else:
        return SmdGroup(
            body=members_sorted[0], marks=members_sorted[1:], group_type="complex"
        )


def _cluster_smds(candidates: list[RectEntity], eps: float = None) -> list[SmdGroup]:
    """
    DBSCAN 空間分群，再對每群做角色分類
    eps 預設為候選矩形平均尺寸的 2 倍
    """
    if not candidates:
        return []

    centers = np.array([[r.cx, r.cy] for r in candidates])

    if eps is None:
        avg_size = np.mean([max(r.bbox.w, r.bbox.h) for r in candidates])
        eps = avg_size * 2.5

    db = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = db.labels_

    area_threshold = _auto_area_threshold(candidates)

    groups = []
    for lbl in set(labels):
        members = [candidates[i] for i, l in enumerate(labels) if l == lbl]
        group = _classify_group(members, area_threshold)
        groups.append(group)

    return groups


# ─────────────────────────────────────────────
# Step 10：SMD 方向分配
# ─────────────────────────────────────────────


def _assign_smd_sides(
    groups: list[SmdGroup], die_bbox: Bbox, is_inner: bool = False
) -> list[SmdGroup]:
    """
    依相對於 die_bbox 中心的方向，分配 top/bottom/left/right
    """
    die_cx, die_cy = die_bbox.cx, die_bbox.cy
    for g in groups:
        cx, cy = g.center
        dx = cx - die_cx
        dy = cy - die_cy
        g.is_inner = is_inner
        if abs(dy) >= abs(dx):
            g.side = "top" if dy > 0 else "bottom"
        else:
            g.side = "right" if dx > 0 else "left"
    return groups


# ─────────────────────────────────────────────
# Fiducial 形狀分類
# ─────────────────────────────────────────────


def _classify_fiducial_shape(entities: list) -> str:
    """
    判斷 fiducial 的形狀類型：
      circle   → 有 CIRCLE entity
      cross    → 多條 LINE 組成（或十字形 LWPOLYLINE）
      unknown  → 其他
    """
    types = [e.dxftype() for e in entities]

    # 有 CIRCLE → circle
    if "CIRCLE" in types:
        return "circle"

    # 全是 LINE → 看數量和方向判斷是否為十字
    lines = [e for e in entities if e.dxftype() == "LINE"]
    if len(lines) >= 2:
        # 十字特徵：有水平線也有垂直線
        has_h = any(abs(l.dxf.start.y - l.dxf.end.y) < 1e-6 for l in lines)
        has_v = any(abs(l.dxf.start.x - l.dxf.end.x) < 1e-6 for l in lines)
        if has_h and has_v:
            return "cross"
        # 只有線段也算 cross（可能是斜線十字）
        return "cross"

    # LWPOLYLINE 判斷
    polys = [e for e in entities if e.dxftype() == "LWPOLYLINE"]
    for p in polys:
        pts = list(p.get_points())
        if len(pts) == 4:
            # 4頂點閉合 → 矩形，可能是 cross 的一部分
            return "cross"

    return "unknown"


def _group_fiducials(fiducials_by_corner: dict) -> list:
    """
    把每個角落的 entities 組成 FiducialGroup，
    分析形狀並依形狀分組。
    回傳 list of (shape, entities_per_corner)
    """
    groups = []
    for corner_idx, entities in fiducials_by_corner.items():
        if not entities:
            continue
        shape = _classify_fiducial_shape(entities)
        groups.append(
            {
                "corner_idx": corner_idx,
                "shape": shape,
                "entities": entities,
            }
        )
    return groups


# ─────────────────────────────────────────────
# SMD signature（長相辨識）
# ─────────────────────────────────────────────


def _smd_signature(group: "SmdGroup") -> tuple:
    """
    計算 SMD 的 signature，相同長相的 SMD 會有相同 signature。

    signature = (entity_count, rounded_area_ratios)

    例：
      body(100) + mark(10) + mark(10)
        → (3, (1.0, 0.10, 0.10))
      mark(10) + mark(10)
        → (2, (1.0, 1.0))
    """
    members = ([group.body] if group.body else []) + group.marks
    if not members:
        return (0,)

    areas = sorted([m.area for m in members], reverse=True)
    max_area = areas[0]
    if max_area == 0:
        return (len(areas),)

    # 面積比例四捨五入到 2 位，避免浮點誤差造成同類型被分開
    ratios = tuple(round(a / max_area, 2) for a in areas)
    return (len(areas), ratios)


def _assign_smd_types(smd_groups: list) -> dict:
    """
    把所有 SMD groups 依 signature 分類，
    回傳 {signature: type_index} 的對應表，
    以及 {group_id: type_index}
    """
    sig_to_type = {}
    type_counter = 0
    group_types = []

    for g in smd_groups:
        sig = _smd_signature(g)
        if sig not in sig_to_type:
            sig_to_type[sig] = type_counter
            type_counter += 1
        group_types.append(sig_to_type[sig])

    return group_types


# ─────────────────────────────────────────────
# Handle 擷取
# ─────────────────────────────────────────────


def _get_handles(entity) -> list[str]:
    """
    取得一個 entity 的所有 handle。
    大多數情況只有一個 handle，
    但如果是 compound entity（INSERT block）則可能有多個。
    """
    try:
        h = entity.dxf.handle
        if h:
            return [h]
    except Exception:
        pass
    return []


def _entities_to_handles(entities: list) -> list[str]:
    """把多個 entities 的 handle 合併成一個 list"""
    handles = []
    for e in entities:
        handles.extend(_get_handles(e))
    return handles


def _rect_to_handles(rect_entity) -> list[str]:
    """從 RectEntity 取得 handles"""
    return _get_handles(rect_entity.entity)


# ─────────────────────────────────────────────
# JSON 輸出
# ─────────────────────────────────────────────


def export_json(packages: list, indent: int = 2) -> str:
    """
    把解析結果轉成指定格式的 JSON 字串。

    格式：
    {
      "frontside.SMD.0": [
          ["h1", "h2", "h3"],   ← SMD instance 0（body+mark+mark 的所有 handle）
          ["h4", "h5", "h6"],   ← SMD instance 1
      ],
      "frontside.SMD.1": [...],
      "frontside.substrate.0": ["h1", "h2", "h3", "h4"],
      "frontside.lid_outer.0": ["h1"],
      "frontside.inner_lid.0": ["h1"],
      "frontside.BGA.0": ["h1", "h2", ...],
      "frontside.Pin-1.0": ["h1"],
      "frontside.fiducial_mark.cross": [["h1","h2","h3","h4"], ["h5"]],
      "frontside.fiducial_mark.circle": [["h6"], ["h7"]],
      "bottomside.SMD.0": [...],
      ...
    }
    """
    import json

    output = {}

    # ── 決定每個 package 的 side ──────────────────────
    # 規則：
    #   有 Pin-1 → frontside（一定）
    #   沒有 Pin-1：
    #     只有一個 package → bottomside（單面圖）
    #     有兩個 package：
    #       找有 Pin-1 的那個的位置，另一個用相對位置判斷
    #       若都沒有 Pin-1 → 位置較高（y大）或較右（x大）的為 frontside

    def _assign_sides(packages: list) -> list[str]:
        if not packages:
            return []

        front_indices = [i for i, p in enumerate(packages) if p.is_front_view]

        if len(packages) == 1:
            return ["frontside" if packages[0].is_front_view else "bottomside"]

        # 多個 package 的情況
        sides = ["bottomside"] * len(packages)

        # 有 Pin-1 的一定是 frontside
        for i in front_indices:
            sides[i] = "frontside"

        # 沒有 Pin-1 的 package，用位置判斷
        unknown = [i for i in range(len(packages)) if i not in front_indices]
        if unknown:
            if front_indices:
                # 有已知 frontside，用相對位置判斷剩餘的
                # （剩餘的都是 bottomside，已設好）
                pass
            else:
                # 全部都沒有 Pin-1，用位置判斷
                # 比較 substrate 中心點：x 較大或 y 較大的為 frontside
                centers = [
                    (i, packages[i].substrate.cx, packages[i].substrate.cy)
                    for i in unknown
                ]
                # 先判斷是左右排還是上下排
                xs = [c[1] for c in centers]
                ys = [c[2] for c in centers]
                x_spread = max(xs) - min(xs)
                y_spread = max(ys) - min(ys)

                if x_spread >= y_spread:
                    # 左右排：x 較大（右側）為 frontside
                    front_i = max(centers, key=lambda c: c[1])[0]
                else:
                    # 上下排：y 較大（上方）為 frontside
                    front_i = max(centers, key=lambda c: c[2])[0]

                sides[front_i] = "frontside"

        return sides

    sides = _assign_sides(packages)

    for pkg, side in zip(packages, sides):
        # ── Substrate ──────────────────────────────
        sub_handles = _entities_to_handles(pkg.substrate_entities)
        if sub_handles:
            output[f"{side}.substrate.0"] = sub_handles

        # ── Lid_outer ──────────────────────────────
        lid_handles = _entities_to_handles(pkg.lid_outer_entities)
        if lid_handles:
            output[f"{side}.lid_outer.0"] = lid_handles

        # ── Inner_lid ──────────────────────────────
        inner_handles = _entities_to_handles(pkg.inner_lid_entities)
        if inner_handles:
            output[f"{side}.inner_lid.0"] = inner_handles

        # ── BGA balls ──────────────────────────────
        bga_handles = _entities_to_handles(pkg.bga_entities)
        if bga_handles:
            output[f"{side}.BGA.0"] = bga_handles

        # ── Pin-1 ──────────────────────────────────
        if pkg.pin1:
            pin_handles = _rect_to_handles(pkg.pin1)
            if pin_handles:
                output[f"{side}.Pin-1.0"] = pin_handles

        # ── Fiducial marks ─────────────────────────
        # 依形狀分組：cross / circle / unknown
        fid_by_shape: dict[str, list[list[str]]] = {}
        for fid in pkg.fiducial_groups:
            shape = fid["shape"]
            handles = _entities_to_handles(fid["entities"])
            if handles:
                fid_by_shape.setdefault(shape, []).append(handles)

        for shape, instances in fid_by_shape.items():
            output[f"{side}.fiducial_mark.{shape}"] = instances

        # ── SMD groups ─────────────────────────────
        # 先算每個 group 的 type index
        smd_type_indices = _assign_smd_types(pkg.smd_groups)

        # 依 type index 收集 instances
        smd_by_type: dict[int, list[list[str]]] = {}
        for group, type_idx in zip(pkg.smd_groups, smd_type_indices):
            # 收集這個 SMD 的所有 entity handles
            all_handles = []
            if group.body:
                all_handles.extend(_rect_to_handles(group.body))
            for mark in group.marks:
                all_handles.extend(_rect_to_handles(mark))

            if all_handles:
                smd_by_type.setdefault(type_idx, []).append(all_handles)

        # 寫入輸出（依 type index 排序）
        for type_idx in sorted(smd_by_type.keys()):
            output[f"{side}.SMD.{type_idx}"] = smd_by_type[type_idx]

    return json.dumps(output, indent=indent, ensure_ascii=False)


def export_json_file(packages: list, output_path: str) -> None:
    """輸出 JSON 到檔案"""
    json_str = export_json(packages)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"JSON 已輸出到: {output_path}")


# ─────────────────────────────────────────────
# 主解析器
# ─────────────────────────────────────────────


class CadPackageParser:
    """
    使用方式：
        parser = CadPackageParser("your_file.dxf")
        results = parser.parse()
        for pkg in results:
            print(pkg.summary())

    過濾參數：
        exclude_layers : 排除這些 layer 的 entity（黑名單）
        include_layers : 只保留這些 layer 的 entity（白名單，與 exclude 擇一用）
        exclude_types  : 排除這些 DXF entity type（如 "TEXT", "DIMENSION"）
        include_types  : 只保留這些 DXF entity type
    """

    def __init__(
        self,
        dxf_path: str,
        min_bga: int = 10,
        max_aspect: float = 1.8,
        exclude_layers: Optional[list[str]] = None,
        include_layers: Optional[list[str]] = None,
        exclude_types: Optional[list[str]] = None,
        include_types: Optional[list[str]] = None,
    ):
        self.dxf_path = dxf_path
        self.min_bga = min_bga
        self.max_aspect = max_aspect
        self.exclude_layers = set(exclude_layers or [])
        self.include_layers = set(include_layers or [])
        self.exclude_types = set(exclude_types or [])
        self.include_types = set(include_types or [])

    def _filter_entities(self, entities: list) -> list:
        """
        依 layer 和 type 過濾 entity 清單。

        Layer 規則（擇一）：
          - include_layers 有值 → 只保留這些 layer
          - exclude_layers 有值 → 排除這些 layer
          - 兩者都有值       → include 優先（先 include 再 exclude）

        Type 規則（同上邏輯）：
          - include_types 有值 → 只保留這些 type
          - exclude_types 有值 → 排除這些 type
        """
        result = []
        for e in entities:
            # ── Layer 過濾 ──────────────────────────
            try:
                layer = e.dxf.layer
            except Exception:
                layer = ""

            if self.include_layers:
                if layer not in self.include_layers:
                    continue
            elif self.exclude_layers:
                if layer in self.exclude_layers:
                    continue

            # ── Type 過濾 ───────────────────────────
            etype = e.dxftype()

            if self.include_types:
                if etype not in self.include_types:
                    continue
            elif self.exclude_types:
                if etype in self.exclude_types:
                    continue

            result.append(e)
        return result

    def list_layers(self) -> list[str]:
        """列出 DXF 中所有 layer 名稱（方便使用者決定要過濾哪些）"""
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()
        layers = set()
        for e in msp:
            try:
                layers.add(e.dxf.layer)
            except Exception:
                pass
        return sorted(layers)

    def list_types(self) -> dict[str, int]:
        """列出 DXF 中所有 entity type 及數量"""
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()
        return dict(Counter(e.dxftype() for e in msp))

    def parse(self) -> list[PackageResult]:
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()

        all_entities = self._filter_entities(list(msp))
        rects = _collect_rects_from(all_entities)
        df_circles = _collect_circles_from(all_entities)

        substrates = _find_substrates(rects, df_circles, self.min_bga, self.max_aspect)

        if not substrates:
            print("[警告] 找不到任何封裝圖實例")
            return []

        results = []
        for sub_rect in substrates:
            pkg = self._process_package(sub_rect, rects, df_circles, all_entities)
            if pkg:
                results.append(pkg)

        return results

    def _process_package(
        self,
        sub_rect: RectEntity,
        all_rects: list[RectEntity],
        df_circles: pd.DataFrame,
        all_entities: list,
    ) -> Optional[PackageResult]:
        substrate = sub_rect.bbox

        # ── Pin-1 ───────────────────────────────────────
        pin1 = _find_pin1(all_rects, substrate)
        pin1_id = id(pin1.entity) if pin1 else None
        is_front_view = pin1 is not None

        # ── BGA balls + Die area ────────────────────────
        bga_df, die_bbox = _find_bga_and_die(df_circles, substrate)
        if bga_df.empty:
            return None
        bga_entities = list(bga_df["entity"].values)

        # ── Cavity ──────────────────────────────────────
        cavity = _detect_cavity(bga_df)

        # ── Lid 層數 ────────────────────────────────────
        lid_outer, lid_outer_ents, inner_lid, inner_lid_ents = _find_lid_layers(
            all_rects, substrate, die_bbox
        )

        # ── Fiducial marks（substrate + lid_outer 四角）──
        fid_raw = _find_fiducials(all_entities, substrate)
        fiducial_ids = {id(e) for corner_ents in fid_raw.values() for e in corner_ents}
        # 把 Pin-1 從 fiducial 中移除（Pin-1 也在右上角附近）
        if pin1_id:
            fiducial_ids.discard(pin1_id)
            for idx in fid_raw:
                fid_raw[idx] = [e for e in fid_raw[idx] if id(e) != pin1_id]
        if lid_outer:
            lid_fid_raw = _find_fiducials(all_entities, lid_outer)
            for corner_ents in lid_fid_raw.values():
                for e in corner_ents:
                    fiducial_ids.add(id(e))
            # 合併到 fid_raw（按角落 index）
            for idx, ents in lid_fid_raw.items():
                for e in ents:
                    if e not in fid_raw[idx]:
                        fid_raw[idx].append(e)

        fiducial_groups = _group_fiducials(fid_raw)

        # ── 外部 SMD ────────────────────────────────────
        outer_boundary = lid_outer or substrate
        exclusion_zone = inner_lid or die_bbox

        # 把 substrate entity、lid entities 的 id 也加入排除集合
        excluded_ids = fiducial_ids | {id(sub_rect.entity)}
        if pin1_id:
            excluded_ids.add(pin1_id)
        for e in lid_outer_ents + inner_lid_ents:
            excluded_ids.add(id(e))

        outer_candidates = _collect_smd_candidates(
            all_rects, outer_boundary, exclusion_zone, excluded_ids, None, substrate
        )
        outer_groups = _cluster_smds(outer_candidates)
        outer_groups = _assign_smd_sides(outer_groups, die_bbox, is_inner=False)

        # ── Cavity 內部 SMD ─────────────────────────────
        inner_groups: list[SmdGroup] = []
        if cavity:
            inner_candidates = _collect_smd_candidates(
                all_rects, cavity, None, excluded_ids, None, substrate
            )
            inner_candidates = [
                r for r in inner_candidates if cavity.contains_point(r.cx, r.cy)
            ]
            inner_groups = _cluster_smds(inner_candidates)
            inner_groups = _assign_smd_sides(inner_groups, cavity, is_inner=True)

        all_smds = outer_groups + inner_groups

        return PackageResult(
            substrate=substrate,
            substrate_entities=[sub_rect.entity],
            lid_outer=lid_outer,
            lid_outer_entities=lid_outer_ents,
            inner_lid=inner_lid,
            inner_lid_entities=inner_lid_ents,
            die_bbox=die_bbox,
            bga_entities=bga_entities,
            cavity=cavity,
            bga_count=len(bga_df),
            is_front_view=is_front_view,
            pin1=pin1,
            fiducial_groups=fiducial_groups,
            smd_groups=all_smds,
        )


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────


def main():
    import sys
    import argparse

    parser_cli = argparse.ArgumentParser(
        description="CAD Package Parser - 解析 DXF 封裝圖"
    )
    parser_cli.add_argument("dxf", help="DXF 檔案路徑")
    parser_cli.add_argument(
        "-o", "--output", help="JSON 輸出路徑（不指定則印到 stdout）", default=None
    )
    parser_cli.add_argument(
        "--summary-only", action="store_true", help="只印 summary，不輸出 JSON"
    )
    args = parser_cli.parse_args()

    pkg_parser = CadPackageParser(args.dxf)
    results = pkg_parser.parse()

    print(f"\n共找到 {len(results)} 個封裝圖實例\n", file=sys.stderr)
    for i, pkg in enumerate(results):
        print(f"[Package {i + 1}]", file=sys.stderr)
        print(pkg.summary(), file=sys.stderr)
        for j, smd in enumerate(pkg.smd_groups):
            cx, cy = smd.center
            print(
                f"  SMD[{j:02d}] {smd.group_type:15s} "
                f"{'inner' if smd.is_inner else 'outer':5s} "
                f"{smd.side:6s} "
                f"center=({cx:.1f}, {cy:.1f})",
                file=sys.stderr,
            )

    if not args.summary_only:
        json_str = export_json(results)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"\nJSON 已輸出到: {args.output}", file=sys.stderr)
        else:
            print(json_str)


if __name__ == "__main__":
    main()
