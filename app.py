from collections import Counter
import math
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, UploadFile, File, Body, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
import ezdxf
from ezdxf.bbox import extents
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.svg import SVGBackend
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely.ops import linemerge

# Initialize FastAPI app
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "templates" / "index.html"

# --- Algorithm constants ---
SNAP_DECIMALS = 6  # Endpoint rounding precision for stable linemerge stitching
SIZE_TOL_RATIO = 0.05  # Relative size tolerance ratio (5%)
SIZE_TOL_MIN = 0.05  # Minimum absolute size tolerance
DIST_TOL_RATIO = 0.05  # Relative distance tolerance ratio (5%)
DIST_TOL_MIN = 0.1  # Minimum absolute distance tolerance
SVG_LINEWEIGHT_SCALING = 0.2  # Make preview SVG strokes thinner
CACHE_TTL_SECONDS = 60 * 60  # 1 hour
MAX_CACHE_SESSIONS = 32
SINGLE_POINT_COUNT_TOL = 2
SINGLE_ASPECT_RATIO_TOL = 0.20  # 20%
SINGLE_DIAG_TOL = 0.10  # 10%
DEFAULT_FLATTEN_TOL = 0.01
FAST_FLATTEN_TOL = 0.08
MAX_EXTRACT_HIGHLIGHTS_RETURN = 250
MAX_HIGHLIGHT_POINTS_RETURN = 400
MAX_EXTRACT_ENTITIES_PREVIEW = 80
MAX_TEMPLATES_PER_CACHE = 16
MAX_SCAN_HIGHLIGHTS_RETURN = 80
MAX_SCAN_MATCHES_WITH_HIGHLIGHTS = 120
CLIENT_TEMPLATE_HIGHLIGHTS_DRAW_DEFAULT = 250
CLIENT_MATCHES_DRAW_DEFAULT = 200
CLIENT_HIGHLIGHTS_PER_MATCH_DRAW_DEFAULT = 80
CLIENT_TOTAL_HIGHLIGHTS_DRAW_DEFAULT = 1800
CLIENT_ENTITIES_PREVIEW_DEFAULT = 60
CLIENT_MATCHES_PREVIEW_DEFAULT = 80
CLIENT_POLYLINE_POINTS_DRAW_DEFAULT = 400
CLICK_PICK_TOL_RATIO = 0.003
CLICK_PICK_TOL_MIN = 0.25
MATCH_SCORE_MAX_DEFAULT = 0.40
MATCH_SCORE_ANCHOR_WEIGHT = 0.30
MATCH_SCORE_SIZE_WEIGHT = 0.45
MATCH_SCORE_DIST_WEIGHT = 0.45
MATCH_SCORE_SHAPE_WEIGHT = 0.10
SCAN_RELEVANT_DXF_TYPES = {
    "CIRCLE",
    "LINE",
    "LWPOLYLINE",
    "ARC",
    "POLYLINE",
    "ELLIPSE",
    "SPLINE",
}
DEFAULT_RUNTIME_CONFIG = {
    "size_tol_ratio": SIZE_TOL_RATIO,
    "size_tol_min": SIZE_TOL_MIN,
    "dist_tol_ratio": DIST_TOL_RATIO,
    "dist_tol_min": DIST_TOL_MIN,
    "single_point_count_tol": SINGLE_POINT_COUNT_TOL,
    "single_aspect_ratio_tol": SINGLE_ASPECT_RATIO_TOL,
    "single_diag_tol": SINGLE_DIAG_TOL,
    "score_max": MATCH_SCORE_MAX_DEFAULT,
    "match_score_anchor_weight": MATCH_SCORE_ANCHOR_WEIGHT,
    "match_score_size_weight": MATCH_SCORE_SIZE_WEIGHT,
    "match_score_dist_weight": MATCH_SCORE_DIST_WEIGHT,
    "match_score_shape_weight": MATCH_SCORE_SHAPE_WEIGHT,
    "extract_highlights_return_limit": MAX_EXTRACT_HIGHLIGHTS_RETURN,
    "extract_entities_preview_limit": MAX_EXTRACT_ENTITIES_PREVIEW,
    "scan_highlights_return_limit": MAX_SCAN_HIGHLIGHTS_RETURN,
    "scan_matches_with_highlights_limit": MAX_SCAN_MATCHES_WITH_HIGHLIGHTS,
    "template_highlights_draw_limit": CLIENT_TEMPLATE_HIGHLIGHTS_DRAW_DEFAULT,
    "matches_draw_limit": CLIENT_MATCHES_DRAW_DEFAULT,
    "highlights_per_match_draw_limit": CLIENT_HIGHLIGHTS_PER_MATCH_DRAW_DEFAULT,
    "total_highlights_draw_limit": CLIENT_TOTAL_HIGHLIGHTS_DRAW_DEFAULT,
    "entities_preview_limit": CLIENT_ENTITIES_PREVIEW_DEFAULT,
    "matches_preview_limit": CLIENT_MATCHES_PREVIEW_DEFAULT,
    "polyline_points_draw_limit": CLIENT_POLYLINE_POINTS_DRAW_DEFAULT,
}
HYPERPARAMETER_SCHEMA = [
    {
        "id": "score_max",
        "label": "Score Max",
        "type": "float",
        "default": MATCH_SCORE_MAX_DEFAULT,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Reject matches whose final score is above this threshold.",
    },
    {
        "id": "size_tol_ratio",
        "label": "Size Tol Ratio",
        "type": "float",
        "default": SIZE_TOL_RATIO,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Relative size tolerance used for entity filtering.",
    },
    {
        "id": "size_tol_min",
        "label": "Size Tol Min",
        "type": "float",
        "default": SIZE_TOL_MIN,
        "min": 0.0,
        "max": 1000.0,
        "step": 0.01,
        "description": "Minimum absolute size tolerance.",
    },
    {
        "id": "dist_tol_ratio",
        "label": "Dist Tol Ratio",
        "type": "float",
        "default": DIST_TOL_RATIO,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Relative distance tolerance used in multi-entity matching.",
    },
    {
        "id": "dist_tol_min",
        "label": "Dist Tol Min",
        "type": "float",
        "default": DIST_TOL_MIN,
        "min": 0.0,
        "max": 1000.0,
        "step": 0.01,
        "description": "Minimum absolute distance tolerance.",
    },
    {
        "id": "single_point_count_tol",
        "label": "Point Count Tol",
        "type": "int",
        "default": SINGLE_POINT_COUNT_TOL,
        "min": 0,
        "max": 100,
        "step": 1,
        "description": "Allowed point-count gap for composite-shape plugin filtering.",
    },
    {
        "id": "single_aspect_ratio_tol",
        "label": "Aspect Ratio Tol",
        "type": "float",
        "default": SINGLE_ASPECT_RATIO_TOL,
        "min": 0.0,
        "max": 5.0,
        "step": 0.01,
        "description": "Allowed relative aspect-ratio gap for composite-shape filtering.",
    },
    {
        "id": "single_diag_tol",
        "label": "Diag Tol",
        "type": "float",
        "default": SINGLE_DIAG_TOL,
        "min": 0.0,
        "max": 5.0,
        "step": 0.01,
        "description": "Allowed relative bounding-diagonal gap for composite-shape filtering.",
    },
    {
        "id": "match_score_anchor_weight",
        "label": "Anchor Weight",
        "type": "float",
        "default": MATCH_SCORE_ANCHOR_WEIGHT,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Weight assigned to the anchor entity score.",
    },
    {
        "id": "match_score_size_weight",
        "label": "Size Weight",
        "type": "float",
        "default": MATCH_SCORE_SIZE_WEIGHT,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Weight assigned to size error in edge scoring.",
    },
    {
        "id": "match_score_dist_weight",
        "label": "Dist Weight",
        "type": "float",
        "default": MATCH_SCORE_DIST_WEIGHT,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Weight assigned to distance error in edge scoring.",
    },
    {
        "id": "match_score_shape_weight",
        "label": "Shape Weight",
        "type": "float",
        "default": MATCH_SCORE_SHAPE_WEIGHT,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Weight assigned to shape-signature error in edge scoring.",
    },
    {
        "id": "extract_highlights_return_limit",
        "label": "Extract Highlights Return",
        "type": "int",
        "default": MAX_EXTRACT_HIGHLIGHTS_RETURN,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum extract highlights returned by the server.",
    },
    {
        "id": "extract_entities_preview_limit",
        "label": "Extract Preview Return",
        "type": "int",
        "default": MAX_EXTRACT_ENTITIES_PREVIEW,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum extract entities preview returned by the server.",
    },
    {
        "id": "scan_highlights_return_limit",
        "label": "Scan Highlights Per Match",
        "type": "int",
        "default": MAX_SCAN_HIGHLIGHTS_RETURN,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum highlights returned per match by the server.",
    },
    {
        "id": "scan_matches_with_highlights_limit",
        "label": "Matches With Highlights",
        "type": "int",
        "default": MAX_SCAN_MATCHES_WITH_HIGHLIGHTS,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Only the first N matches keep server highlight payloads.",
    },
    {
        "id": "template_highlights_draw_limit",
        "label": "Template Highlights Draw",
        "type": "int",
        "default": CLIENT_TEMPLATE_HIGHLIGHTS_DRAW_DEFAULT,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum template highlights drawn on the client.",
    },
    {
        "id": "matches_draw_limit",
        "label": "Matches Draw",
        "type": "int",
        "default": CLIENT_MATCHES_DRAW_DEFAULT,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum matches drawn on the client.",
    },
    {
        "id": "highlights_per_match_draw_limit",
        "label": "Highlights Per Match Draw",
        "type": "int",
        "default": CLIENT_HIGHLIGHTS_PER_MATCH_DRAW_DEFAULT,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum highlights drawn for each match on the client.",
    },
    {
        "id": "total_highlights_draw_limit",
        "label": "Total Highlights Draw",
        "type": "int",
        "default": CLIENT_TOTAL_HIGHLIGHTS_DRAW_DEFAULT,
        "min": 1,
        "max": 20000,
        "step": 1,
        "description": "Global cap for rendered match highlights on the client.",
    },
    {
        "id": "entities_preview_limit",
        "label": "Entities Preview",
        "type": "int",
        "default": CLIENT_ENTITIES_PREVIEW_DEFAULT,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum entities shown in extract preview on the client.",
    },
    {
        "id": "matches_preview_limit",
        "label": "Matches Preview",
        "type": "int",
        "default": CLIENT_MATCHES_PREVIEW_DEFAULT,
        "min": 1,
        "max": 5000,
        "step": 1,
        "description": "Maximum matches shown in scan preview on the client.",
    },
    {
        "id": "polyline_points_draw_limit",
        "label": "Polyline Points Draw",
        "type": "int",
        "default": CLIENT_POLYLINE_POINTS_DRAW_DEFAULT,
        "min": 2,
        "max": 10000,
        "step": 1,
        "description": "Maximum points kept when drawing long polylines on the client.",
    },
]


def snap(val):
    """Round coordinates to fixed precision for stable linemerge endpoints."""
    return round(val, SNAP_DECIMALS)


def _parse_keep_types(raw_keep_types):
    if not raw_keep_types:
        return set(SCAN_RELEVANT_DXF_TYPES)
    return {
        part.strip().upper() for part in str(raw_keep_types).split(",") if part.strip()
    } or set(SCAN_RELEVANT_DXF_TYPES)


def _prune_entities_by_type(msp, keep_types):
    """Remove DXF entities not present in the keep-types set."""
    removed_count = 0
    for entity in list(msp):
        if entity.dxftype().upper() in keep_types:
            continue
        try:
            msp.delete_entity(entity)
            removed_count += 1
        except Exception:
            pass
    return removed_count


def _remove_top_circle_radii_groups(msp, top_n):
    """Remove CIRCLE entities belonging to the top-N most common radii groups."""
    try:
        n = int(top_n)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return {"removed_count": 0, "removed_groups": []}

    radius_buckets = {}
    for entity in list(msp):
        if entity.dxftype().upper() != "CIRCLE":
            continue
        radius = round(float(entity.dxf.radius), 6)
        radius_buckets.setdefault(radius, []).append(entity)

    if not radius_buckets:
        return {"removed_count": 0, "removed_groups": []}

    ranked = sorted(
        radius_buckets.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    selected = ranked[:n]

    removed_count = 0
    removed_groups = []
    for radius, entities in selected:
        group_removed = 0
        for entity in entities:
            try:
                msp.delete_entity(entity)
                group_removed += 1
            except Exception:
                pass
        if group_removed > 0:
            removed_count += group_removed
            removed_groups.append(
                {
                    "radius": radius,
                    "count": group_removed,
                }
            )

    return {
        "removed_count": removed_count,
        "removed_groups": removed_groups,
    }


# --- Pydantic models (normalized geometry fingerprint) ---
class EntityModel(BaseModel):
    type: str  # CIRCLE, COMPOSITE_SHAPE
    size: float  # Radius or total length/perimeter
    x: float  # Geometry center X
    y: float  # Geometry center Y


class TemplateModel(BaseModel):
    entities: list[EntityModel]


def _coerce_runtime_config(raw):
    config = dict(DEFAULT_RUNTIME_CONFIG)
    if not isinstance(raw, dict):
        return config
    for meta in HYPERPARAMETER_SCHEMA:
        key = meta["id"]
        if key not in raw:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            continue
        if meta["type"] == "int":
            value = int(round(value))
        if "min" in meta:
            value = max(meta["min"], value)
        if "max" in meta:
            value = min(meta["max"], value)
        if meta["type"] == "int":
            value = int(value)
        config[key] = value
    return config


# --- Geometry helper functions ---
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def geometry_to_render_pct(geom, bounds):
    """Convert DXF geometry coordinates to SVG render percentage coordinates."""
    w, h = bounds["width"], bounds["height"]
    min_x, max_y = bounds["min_x"], bounds["max_y"]
    if geom["kind"] == "circle":
        return {
            "kind": "circle",
            "cx_pct": (geom["cx"] - min_x) / w,
            "cy_pct": (max_y - geom["cy"]) / h,
            "r_pct": geom["r"] / w,
        }
    else:  # polyline
        return {
            "kind": "polyline",
            "points_pct": [
                [(px - min_x) / w, (max_y - py) / h] for px, py in geom["points"]
            ],
        }


def _sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _compact_render_highlight(h):
    if h.get("kind") != "polyline":
        return h
    points = h.get("points_pct", [])
    compact = dict(h)
    compact["points_pct"] = _sample_points(points, MAX_HIGHLIGHT_POINTS_RETURN)
    return compact


def get_dxf_bounds(msp):
    """Get DXF extents and dimensions."""
    ext = extents(msp)
    return {
        "min_x": ext.extmin.x,
        "min_y": ext.extmin.y,
        "max_x": ext.extmax.x,
        "max_y": ext.extmax.y,
        "width": ext.extmax.x - ext.extmin.x,
        "height": ext.extmax.y - ext.extmin.y,
    }


def _render_pct_to_dxf_point(pt, bounds):
    return (
        bounds["min_x"] + pt[0] * bounds["width"],
        bounds["max_y"] - pt[1] * bounds["height"],
    )


def _feature_pick_distance(feature, pick_point):
    geom = feature.get("geometry") or {}
    kind = geom.get("kind")
    if kind == "circle":
        center_dist = calculate_distance(
            (geom["cx"], geom["cy"]), (pick_point.x, pick_point.y)
        )
        # Treat the filled interior as a valid hit so click selection feels forgiving.
        return 0.0 if center_dist <= geom["r"] else abs(center_dist - geom["r"])
    if kind == "polyline":
        try:
            return LineString(geom.get("points", [])).distance(pick_point)
        except Exception:
            pass
    return Point(feature["x"], feature["y"]).distance(pick_point)


def _extract_entities_from_polygon(cache_payload, polygon_pct):
    if len(polygon_pct) < 3:
        return []
    bounds = cache_payload["bounds"]
    dxf_vertices = [_render_pct_to_dxf_point(pt, bounds) for pt in polygon_pct]
    roi_polygon = ShapelyPolygon(dxf_vertices)
    return [
        f
        for f in cache_payload["fingerprints"]
        if roi_polygon.covers(Point(f["x"], f["y"]))
    ]


def _extract_entities_from_click(cache_payload, click_pct):
    if not isinstance(click_pct, list) or len(click_pct) < 2:
        return []
    bounds = cache_payload["bounds"]
    pick_point = Point(*_render_pct_to_dxf_point(click_pct, bounds))
    pick_tol = max(
        CLICK_PICK_TOL_MIN,
        max(bounds["width"], bounds["height"]) * CLICK_PICK_TOL_RATIO,
    )

    best_feature = None
    best_distance = float("inf")
    best_center_distance = float("inf")
    for feature in cache_payload["fingerprints"]:
        dist = _feature_pick_distance(feature, pick_point)
        center_dist = Point(feature["x"], feature["y"]).distance(pick_point)
        if dist < best_distance or (
            math.isclose(dist, best_distance) and center_dist < best_center_distance
        ):
            best_feature = feature
            best_distance = dist
            best_center_distance = center_dist

    if best_feature is None or best_distance > pick_tol:
        return []
    return [best_feature]


def _build_extract_response(
    cache_id, bounds, entities_found, selector_mode, runtime_config=None
):
    config = runtime_config or DEFAULT_RUNTIME_CONFIG
    highlight_limit = int(config["extract_highlights_return_limit"])
    preview_limit = int(config["extract_entities_preview_limit"])
    if not entities_found:
        return {
            "selector_mode": selector_mode,
            "entities": [],
            "highlights": [],
        }

    group_cx = sum(e["x"] for e in entities_found) / len(entities_found)
    group_cy = sum(e["y"] for e in entities_found) / len(entities_found)

    entities_clean = [
        {k: v for k, v in e.items() if k != "geometry"} for e in entities_found
    ]
    template_payload = {
        "group_center": {"x": round(group_cx, 3), "y": round(group_cy, 3)},
        "entities": entities_clean,
    }
    template_id = _store_extracted_template(
        cache_id, template_payload["group_center"], entities_clean
    )
    all_highlights = [
        geometry_to_render_pct(e["geometry"], bounds) for e in entities_found
    ]
    highlights = [
        _compact_render_highlight(h) for h in all_highlights[:highlight_limit]
    ]

    return {
        "cache_id": cache_id,
        "template_id": template_id,
        "selector_mode": selector_mode,
        "group_center": template_payload["group_center"],
        "entity_count": len(entities_clean),
        "entities_preview": entities_clean[:preview_limit],
        "highlights": highlights,
        "highlight_count_total": len(all_highlights),
        "highlight_count_returned": len(highlights),
        "highlight_truncated": len(all_highlights) > len(highlights),
    }


def _polyline_shape_signature(points):
    """Build an orientation-agnostic shape signature for polyline matching."""
    if not points:
        return {"point_count": 0, "bbox_diag": 0.0, "aspect_ratio": 1.0}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    long_side = max(w, h)
    short_side = max(min(w, h), 1e-9)
    return {
        "point_count": int(len(points)),
        "bbox_diag": round(math.hypot(w, h), 6),
        "aspect_ratio": round(long_side / short_side, 6),
    }


def extract_template_features(
    msp, roi_box=None, *, flatten_tol=DEFAULT_FLATTEN_TOL, skip_ellipse_spline=False
):
    """
    Smart feature extraction v2:
    1. Always run global linemerge on the whole drawing for topology consistency.
    2. Supports CIRCLE, LINE, LWPOLYLINE, ARC, POLYLINE, ELLIPSE, SPLINE.
    3. Snap endpoints to improve linemerge stability.
    4. Apply ROI filtering last (centroid must be inside ROI).
    """
    circles = []
    all_lines = []

    for entity in msp:
        etype = entity.dxftype()

        if etype == "CIRCLE":
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            circles.append(
                {
                    "type": "CIRCLE",
                    "size": round(entity.dxf.radius, 3),
                    "x": round(cx, 3),
                    "y": round(cy, 3),
                    "geometry": {
                        "kind": "circle",
                        "cx": cx,
                        "cy": cy,
                        "r": entity.dxf.radius,
                    },
                }
            )

        elif etype == "LINE":
            p1, p2 = entity.dxf.start, entity.dxf.end
            all_lines.append(
                LineString([(snap(p1.x), snap(p1.y)), (snap(p2.x), snap(p2.y))])
            )

        elif etype == "LWPOLYLINE":
            pts = [(snap(p[0]), snap(p[1])) for p in entity.get_points()]
            if entity.is_closed and len(pts) >= 3:
                pts.append(pts[0])
            if len(pts) >= 2:
                all_lines.append(LineString(pts))

        elif etype == "ARC":
            center = entity.dxf.center
            radius = entity.dxf.radius
            sa = math.radians(entity.dxf.start_angle)
            ea = math.radians(entity.dxf.end_angle)
            if ea <= sa:
                ea += 2 * math.pi
            n = max(8, int(abs(ea - sa) / (math.pi / 18)))
            pts = [
                (
                    snap(center.x + radius * math.cos(sa + (ea - sa) * i / n)),
                    snap(center.y + radius * math.sin(sa + (ea - sa) * i / n)),
                )
                for i in range(n + 1)
            ]
            all_lines.append(LineString(pts))

        elif etype == "POLYLINE":
            try:
                pts = [
                    (snap(v.dxf.location.x), snap(v.dxf.location.y))
                    for v in entity.vertices
                ]
                if entity.is_closed and len(pts) >= 3:
                    pts.append(pts[0])
                if len(pts) >= 2:
                    all_lines.append(LineString(pts))
            except Exception:
                pass

        elif etype in ("ELLIPSE", "SPLINE"):
            if skip_ellipse_spline:
                continue
            try:
                pts = [(snap(p.x), snap(p.y)) for p in entity.flattening(flatten_tol)]
                if len(pts) >= 2:
                    all_lines.append(LineString(pts))
            except Exception:
                pass

    # --- Global linemerge: always merge the full drawing for consistent topology ---
    features = list(circles)
    if all_lines:
        merged_result = linemerge(all_lines)

        if merged_result.geom_type in ("LineString", "LinearRing"):
            merged_geoms = [merged_result]
        else:
            merged_geoms = list(merged_result.geoms)

        for geom in merged_geoms:
            centroid = geom.centroid
            features.append(
                {
                    "type": "COMPOSITE_SHAPE",
                    "size": round(geom.length, 3),
                    "x": round(centroid.x, 3),
                    "y": round(centroid.y, 3),
                    "shape_sig": _polyline_shape_signature(list(geom.coords)),
                    "geometry": {
                        "kind": "polyline",
                        "points": [list(c) for c in geom.coords],
                    },
                }
            )

    # --- ROI filtering: keep features whose centroids are inside ROI ---
    if roi_box is not None:
        features = [f for f in features if roi_box.covers(Point(f["x"], f["y"]))]

    return features


# --- Session cache store ---
_cache_store = {}
_cache_lock = threading.Lock()


def _build_cache_payload(msp, *, fast_build=False):
    """Build cache payload (fingerprints + spatial indexes) for one upload."""
    bounds = get_dxf_bounds(msp)
    flatten_tol = FAST_FLATTEN_TOL if fast_build else DEFAULT_FLATTEN_TOL
    fingerprints = extract_template_features(
        msp,
        roi_box=None,
        flatten_tol=flatten_tol,
        skip_ellipse_spline=fast_build,
    )

    n = len(fingerprints)
    if n > 0:
        coords = np.array([[f["x"], f["y"]] for f in fingerprints])
        sizes = np.array([f["size"] for f in fingerprints])
        tree = cKDTree(coords)
    else:
        coords = np.empty((0, 2))
        sizes = np.empty(0)
        tree = None

    # Build per-type numpy indexes for fast pre-filtering
    type_index = {}
    for idx, f in enumerate(fingerprints):
        type_index.setdefault(f["type"], []).append(idx)
    type_index = {t: np.array(v) for t, v in type_index.items()}

    now = time.time()
    return {
        "bounds": bounds,
        "fingerprints": fingerprints,
        "coords": coords,
        "sizes": sizes,
        "tree": tree,
        "type_index": type_index,
        "templates": {},
        "fast_build": fast_build,
        "flatten_tol": flatten_tol,
        "created_at": now,
        "last_access": now,
    }


def _feature_identity_key(feature):
    shape_sig = feature.get("shape_sig")
    shape_key = None
    if isinstance(shape_sig, dict):
        shape_key = (
            shape_sig.get("point_count"),
            shape_sig.get("bbox_diag"),
            shape_sig.get("aspect_ratio"),
        )
    return (
        feature.get("type"),
        round(float(feature.get("size", 0.0)), 6),
        round(float(feature.get("x", 0.0)), 6),
        round(float(feature.get("y", 0.0)), 6),
        shape_key,
    )


def _refresh_cache_indexes(payload):
    """Rebuild numpy/KD-tree indexes after cache mutations."""
    fingerprints = payload.get("fingerprints", [])
    n = len(fingerprints)
    if n > 0:
        coords = np.array([[f["x"], f["y"]] for f in fingerprints])
        sizes = np.array([f["size"] for f in fingerprints])
        tree = cKDTree(coords)
    else:
        coords = np.empty((0, 2))
        sizes = np.empty(0)
        tree = None

    type_index = {}
    for idx, f in enumerate(fingerprints):
        type_index.setdefault(f["type"], []).append(idx)

    payload["coords"] = coords
    payload["sizes"] = sizes
    payload["tree"] = tree
    payload["type_index"] = {t: np.array(v) for t, v in type_index.items()}


def _prune_cache_locked(now):
    expired = [
        cid
        for cid, payload in _cache_store.items()
        if now - payload["last_access"] > CACHE_TTL_SECONDS
    ]
    for cid in expired:
        _cache_store.pop(cid, None)

    if len(_cache_store) <= MAX_CACHE_SESSIONS:
        return

    overflow = len(_cache_store) - MAX_CACHE_SESSIONS
    oldest = sorted(_cache_store.items(), key=lambda item: item[1]["last_access"])
    for cid, _ in oldest[:overflow]:
        _cache_store.pop(cid, None)


def _store_cache(cache_payload):
    """Store cache payload and return its cache_id."""
    cache_id = uuid.uuid4().hex
    now = time.time()
    with _cache_lock:
        cache_payload["last_access"] = now
        _cache_store[cache_id] = cache_payload
        _prune_cache_locked(now)
    return cache_id


def _get_cache(cache_id):
    """Fetch cache payload by id, update access timestamp, return None if missing."""
    if not cache_id:
        return None
    now = time.time()
    with _cache_lock:
        _prune_cache_locked(now)
        payload = _cache_store.get(cache_id)
        if payload is not None:
            payload["last_access"] = now
        return payload


def _store_extracted_template(cache_id, group_center, entities):
    """Store extracted template server-side and return template_id."""
    if not cache_id:
        return None
    now = time.time()
    with _cache_lock:
        payload = _cache_store.get(cache_id)
        if payload is None:
            return None
        templates = payload.setdefault("templates", {})
        template_id = uuid.uuid4().hex
        templates[template_id] = {
            "group_center": group_center,
            "entities": entities,
            "created_at": now,
        }
        # Keep only most recent templates to bound memory.
        if len(templates) > MAX_TEMPLATES_PER_CACHE:
            oldest = sorted(
                templates.items(), key=lambda item: item[1].get("created_at", 0)
            )
            for tid, _ in oldest[: len(templates) - MAX_TEMPLATES_PER_CACHE]:
                templates.pop(tid, None)
        payload["last_access"] = now
        return template_id


def _load_extracted_template(cache_id, template_id):
    """Load extracted template from cache by ids."""
    if not cache_id or not template_id:
        return None
    now = time.time()
    with _cache_lock:
        payload = _cache_store.get(cache_id)
        if payload is None:
            return None
        payload["last_access"] = now
        return payload.get("templates", {}).get(template_id)


def _remove_entities_from_cache(cache_id, entities):
    """Remove selected fingerprint entities from cache and rebuild indexes."""
    if not cache_id or not entities:
        return None, []

    remove_keys = {_feature_identity_key(entity) for entity in entities}
    now = time.time()
    with _cache_lock:
        payload = _cache_store.get(cache_id)
        if payload is None:
            return None, []

        removed = []
        kept = []
        for feature in payload.get("fingerprints", []):
            if _feature_identity_key(feature) in remove_keys:
                removed.append(feature)
            else:
                kept.append(feature)

        if not removed:
            payload["last_access"] = now
            return payload, []

        payload["fingerprints"] = kept
        payload["templates"] = {}
        payload["last_access"] = now
        _refresh_cache_indexes(payload)
        return payload, removed


def _match_selected_features(payload, entities):
    """Resolve selected lightweight entities back to cached fingerprint features."""
    buckets = {}
    for feature in payload.get("fingerprints", []):
        buckets.setdefault(_feature_identity_key(feature), []).append(feature)

    selected = []
    counts = Counter(_feature_identity_key(entity) for entity in entities)
    for key, wanted in counts.items():
        matches = buckets.get(key, [])
        selected.extend(matches[:wanted])
    return selected


def _build_selected_dxf(selected_features):
    """Create a DXF document containing only the selected fingerprint geometries."""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    for feature in selected_features:
        geom = feature.get("geometry") or {}
        kind = geom.get("kind")
        if kind == "circle":
            msp.add_circle((geom["cx"], geom["cy"]), geom["r"])
            continue

        if kind != "polyline":
            continue

        points = [tuple(pt) for pt in geom.get("points", [])]
        if len(points) < 2:
            continue
        if len(points) == 2:
            msp.add_line(points[0], points[1])
            continue

        is_closed = len(points) >= 4 and points[0] == points[-1]
        lw_points = points[:-1] if is_closed else points
        if len(lw_points) >= 2:
            msp.add_lwpolyline(lw_points, close=is_closed)
    return doc


def _render_pct_to_dxf_geom(highlight, bounds):
    kind = highlight.get("kind")
    if kind == "circle":
        return {
            "kind": "circle",
            "cx": bounds["min_x"] + highlight["cx_pct"] * bounds["width"],
            "cy": bounds["max_y"] - highlight["cy_pct"] * bounds["height"],
            "r": highlight["r_pct"] * bounds["width"],
        }
    if kind == "polyline":
        return {
            "kind": "polyline",
            "points": [
                (
                    bounds["min_x"] + pt[0] * bounds["width"],
                    bounds["max_y"] - pt[1] * bounds["height"],
                )
                for pt in highlight.get("points_pct", [])
            ],
        }
    return None


def _build_matches_dxf(matches, bounds):
    """Create a DXF document containing match highlight geometries."""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    exported = 0
    for match in matches:
        for highlight in match.get("highlights", []) or []:
            geom = _render_pct_to_dxf_geom(highlight, bounds)
            if not geom:
                continue
            if geom["kind"] == "circle":
                msp.add_circle((geom["cx"], geom["cy"]), geom["r"])
                exported += 1
                continue

            points = geom.get("points", [])
            if len(points) < 2:
                continue
            if len(points) == 2:
                msp.add_line(points[0], points[1])
                exported += 1
                continue

            is_closed = len(points) >= 4 and points[0] == points[-1]
            lw_points = points[:-1] if is_closed else points
            if len(lw_points) >= 2:
                msp.add_lwpolyline(lw_points, close=is_closed)
                exported += 1
    return doc, exported


def _find_matching(adj):
    """Kuhn's algorithm for maximum bipartite matching.
    adj[i] = list of right-node indices that left node i can match to.
    Returns matched right index set (perfect matching) or None.
    """
    n = len(adj)
    if n == 0:
        return set()
    match_r = {}  # right -> left

    def _augment(u, visited):
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if v not in match_r or _augment(match_r[v], visited):
                    match_r[v] = u
                    return True
        return False

    # Most-constrained-first: match target with fewest candidates first
    order = sorted(range(n), key=lambda i: len(adj[i]))
    for u in order:
        if not _augment(u, set()):
            return None  # This node cannot be matched -> no perfect matching
    return set(match_r.keys())


def _pick_anchor(entities, all_fp, type_index, sizes, *, config, enabled_plugins):
    """Pick the entity with the fewest global candidates as anchor (rarest first)."""
    best_idx, best_cnt = 0, float("inf")
    for i, e in enumerate(entities):
        t = e["type"]
        if t not in type_index:
            return i  # Type not found globally -> 0 candidates; fast fail
        tol = max(config["size_tol_min"], e["size"] * config["size_tol_ratio"])
        base_idx = type_index[t][np.abs(sizes[type_index[t]] - e["size"]) < tol]
        if len(base_idx) == 0:
            return i
        cnt = sum(
            1
            for idx in base_idx
            if _single_entity_plugin_pass(e, all_fp[int(idx)], config, enabled_plugins)
        )
        if cnt < best_cnt:
            best_cnt = cnt
            best_idx = i
    return best_idx


SingleEntityPlugin = Callable[[dict, dict, dict], bool]
SINGLE_ENTITY_MATCH_PLUGINS: dict[str, dict] = {}


def register_single_entity_plugin(
    plugin_id: str,
    label: str,
    description: str,
    plugin: SingleEntityPlugin,
    *,
    default_enabled: bool = True,
):
    """Register a single-entity matching plugin."""
    SINGLE_ENTITY_MATCH_PLUGINS[plugin_id] = {
        "id": plugin_id,
        "label": label,
        "description": description,
        "default_enabled": bool(default_enabled),
        "fn": plugin,
    }


def _resolve_enabled_plugins(raw_plugins):
    defaults = {
        pid
        for pid, meta in SINGLE_ENTITY_MATCH_PLUGINS.items()
        if meta.get("default_enabled")
    }
    if raw_plugins is None:
        return defaults
    if not isinstance(raw_plugins, list):
        return defaults
    resolved = {
        str(pid) for pid in raw_plugins if str(pid) in SINGLE_ENTITY_MATCH_PLUGINS
    }
    return resolved


def _plugin_composite_shape_signature(template_entity, candidate_feature, config):
    """Filter COMPOSITE_SHAPE candidates using extra shape signature fields."""
    if template_entity.get("type") != "COMPOSITE_SHAPE":
        return True
    ts = template_entity.get("shape_sig")
    cs = candidate_feature.get("shape_sig")
    if not isinstance(ts, dict) or not isinstance(cs, dict):
        return True

    tp = ts.get("point_count")
    cp = cs.get("point_count")
    if isinstance(tp, (int, float)) and isinstance(cp, (int, float)):
        if abs(int(tp) - int(cp)) > int(config["single_point_count_tol"]):
            return False

    ta = ts.get("aspect_ratio")
    ca = cs.get("aspect_ratio")
    if isinstance(ta, (int, float)) and isinstance(ca, (int, float)) and ta > 0:
        if abs(ca - ta) / ta > config["single_aspect_ratio_tol"]:
            return False

    td = ts.get("bbox_diag")
    cd = cs.get("bbox_diag")
    if isinstance(td, (int, float)) and isinstance(cd, (int, float)) and td > 0:
        if abs(cd - td) / td > config["single_diag_tol"]:
            return False
    return True


def _single_entity_plugin_pass(
    template_entity,
    candidate_feature,
    config,
    enabled_plugins,
):
    for plugin_id in enabled_plugins:
        meta = SINGLE_ENTITY_MATCH_PLUGINS.get(plugin_id)
        if meta is None:
            continue
        if not meta["fn"](template_entity, candidate_feature, config):
            return False
    return True


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def _shape_signature_error(template_entity, candidate_feature, config):
    """Normalized shape signature error (0..1), only for COMPOSITE_SHAPE."""
    if template_entity.get("type") != "COMPOSITE_SHAPE":
        return 0.0
    ts = template_entity.get("shape_sig")
    cs = candidate_feature.get("shape_sig")
    if not isinstance(ts, dict) or not isinstance(cs, dict):
        return 0.0

    parts = []

    tp = ts.get("point_count")
    cp = cs.get("point_count")
    if isinstance(tp, (int, float)) and isinstance(cp, (int, float)):
        parts.append(
            _clamp01(
                abs(int(tp) - int(cp)) / max(1, int(config["single_point_count_tol"]))
            )
        )

    ta = ts.get("aspect_ratio")
    ca = cs.get("aspect_ratio")
    if isinstance(ta, (int, float)) and isinstance(ca, (int, float)) and ta > 0:
        rel = abs(ca - ta) / ta
        parts.append(_clamp01(rel / max(config["single_aspect_ratio_tol"], 1e-9)))

    td = ts.get("bbox_diag")
    cd = cs.get("bbox_diag")
    if isinstance(td, (int, float)) and isinstance(cd, (int, float)) and td > 0:
        rel = abs(cd - td) / td
        parts.append(_clamp01(rel / max(config["single_diag_tol"], 1e-9)))

    if not parts:
        return 0.0
    return float(sum(parts) / len(parts))


def _pair_match_score(
    template_entity,
    candidate_feature,
    *,
    size_tol,
    config,
    dist_error_ratio=None,
):
    """Compute normalized pair match score (lower is better)."""
    size_err = _clamp01(
        abs(candidate_feature["size"] - template_entity["size"]) / size_tol
    )
    shape_err = _shape_signature_error(template_entity, candidate_feature, config)
    if dist_error_ratio is None:
        return round(0.85 * size_err + 0.15 * shape_err, 6)
    dist_err = _clamp01(dist_error_ratio)
    return round(
        config["match_score_size_weight"] * size_err
        + config["match_score_dist_weight"] * dist_err
        + config["match_score_shape_weight"] * shape_err,
        6,
    )


def _resolve_score_max(template, config):
    raw = template.get("score_max", config["score_max"])
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return config["score_max"]
    return _clamp01(v)


def _entity_basic_match(
    template_entity,
    candidate_feature,
    size_tol=None,
    *,
    config,
    enabled_plugins,
):
    """Base entity matching: type + size (+ optional shape plugins)."""
    if candidate_feature.get("type") != template_entity.get("type"):
        return False
    tol = (
        size_tol
        if size_tol is not None
        else max(
            config["size_tol_min"], template_entity["size"] * config["size_tol_ratio"]
        )
    )
    if abs(candidate_feature.get("size", 0.0) - template_entity["size"]) >= tol:
        return False
    return _single_entity_plugin_pass(
        template_entity, candidate_feature, config, enabled_plugins
    )


def _single_entity_matches(all_fp, anchor_tmpl, *, config, enabled_plugins):
    """Find single-entity matches by base filters + pluggable plugin filters."""
    return [
        i
        for i, f in enumerate(all_fp)
        if _entity_basic_match(
            anchor_tmpl,
            f,
            config=config,
            enabled_plugins=enabled_plugins,
        )
    ]


def _best_scored_matching(adj_scored):
    """Solve min-cost one-to-one matching from scored adjacency lists."""
    n = len(adj_scored)
    if n == 0:
        return set(), 0.0

    right_nodes = sorted({idx for row in adj_scored for idx, _ in row})
    if len(right_nodes) < n:
        return None

    jmap = {rid: j for j, rid in enumerate(right_nodes)}
    inf_cost = 1e6
    cost = np.full((n, len(right_nodes)), inf_cost, dtype=np.float64)
    for i, row in enumerate(adj_scored):
        for rid, s in row:
            j = jmap[rid]
            if s < cost[i, j]:
                cost[i, j] = s

    row_ind, col_ind = linear_sum_assignment(cost)
    if len(row_ind) != n:
        return None
    selected = cost[row_ind, col_ind]
    if np.any(selected >= inf_cost):
        return None

    matched = {right_nodes[int(j)] for j in col_ind}
    return matched, float(np.mean(selected))


register_single_entity_plugin(
    "composite_shape_signature",
    "Composite Shape Signature",
    "Filter composite-shape candidates by point count, aspect ratio, and bbox diagonal.",
    _plugin_composite_shape_signature,
    default_enabled=True,
)

# --- FastAPI routes ---


@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")


@app.get("/settings_schema")
async def get_settings_schema():
    return {
        "plugins": [
            {
                "id": meta["id"],
                "label": meta["label"],
                "description": meta["description"],
                "default_enabled": meta["default_enabled"],
            }
            for meta in SINGLE_ENTITY_MATCH_PLUGINS.values()
        ],
        "hyperparameters": HYPERPARAMETER_SCHEMA,
    }


@app.post("/upload")
async def upload_dxf(
    file: UploadFile = File(...),
    fast_build: bool = Form(False),
    prune_entities: bool = Form(False),
    keep_types: str = Form(""),
    remove_top_circle_radii: int = Form(0),
):
    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Empty upload"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        doc = ezdxf.readfile(tmp_path)
    except Exception as exc:
        return JSONResponse(
            status_code=400, content={"error": f"Failed to read DXF: {str(exc)}"}
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    msp = doc.modelspace()
    dxf_entity_count = len(msp)
    removed_entity_count = 0
    removed_circle_count = 0
    removed_circle_groups = []
    keep_types_set = _parse_keep_types(keep_types)
    if prune_entities:
        removed_entity_count = _prune_entities_by_type(msp, keep_types_set)
    circle_prune_result = _remove_top_circle_radii_groups(msp, remove_top_circle_radii)
    removed_circle_count = circle_prune_result["removed_count"]
    removed_circle_groups = circle_prune_result["removed_groups"]
    retained_dxf_entity_count = len(msp)
    backend = SVGBackend()
    render_config = Configuration.defaults().with_changes(
        lineweight_scaling=SVG_LINEWEIGHT_SCALING
    )

    t0 = time.perf_counter()
    Frontend(RenderContext(doc), backend, config=render_config).draw_layout(msp)
    page = layout.Page(0, 0, layout.Units.mm)
    svg = backend.get_string(page)
    svg_render_time_ms = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    cache_payload = _build_cache_payload(msp, fast_build=fast_build)
    cache_build_time_ms = int((time.perf_counter() - t1) * 1000)
    cache_id = _store_cache(cache_payload)
    return {
        "status": "success",
        "cache_id": cache_id,
        "build_mode": "fast" if fast_build else "accurate",
        "prune_entities": prune_entities,
        "keep_types": sorted(keep_types_set),
        "remove_top_circle_radii": max(0, int(remove_top_circle_radii or 0)),
        "dxf_entity_count": dxf_entity_count,
        "retained_dxf_entity_count": retained_dxf_entity_count,
        "removed_entity_count": removed_entity_count,
        "removed_circle_count": removed_circle_count,
        "removed_circle_groups": removed_circle_groups,
        "entity_count": len(cache_payload["fingerprints"]),
        "svg_render_time_ms": svg_render_time_ms,
        "cache_build_time_ms": cache_build_time_ms,
        "svg": svg,
    }


@app.post("/extract")
async def extract_template(data: dict = Body(...)):
    cache_id = data.get("cache_id")
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    polygon_pct = data.get("polygon_pct", [])
    click_pct = data.get("click_pct")
    runtime_config = _coerce_runtime_config(data.get("settings"))
    bounds = cache_payload["bounds"]
    if isinstance(click_pct, list) and len(click_pct) >= 2:
        entities_found = _extract_entities_from_click(cache_payload, click_pct)
        return _build_extract_response(
            cache_id,
            bounds,
            entities_found,
            "click",
            runtime_config=runtime_config,
        )

    entities_found = _extract_entities_from_polygon(cache_payload, polygon_pct)
    return _build_extract_response(
        cache_id,
        bounds,
        entities_found,
        "polygon",
        runtime_config=runtime_config,
    )


@app.post("/template")
async def get_template(data: dict = Body(...)):
    cache_id = data.get("cache_id")
    template_id = data.get("template_id")
    template = _load_extracted_template(cache_id, template_id)
    if template is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Template not found. Please extract template again."},
        )
    return {
        "cache_id": cache_id,
        "template_id": template_id,
        "group_center": template["group_center"],
        "entities": template["entities"],
    }


@app.post("/remove_entities")
async def remove_entities(data: dict = Body(...)):
    cache_id = data.get("cache_id")
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    template_id = data.get("template_id")
    if template_id:
        stored_template = _load_extracted_template(cache_id, template_id)
        if stored_template is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found. Please extract template again."},
            )
        entities = stored_template.get("entities", [])
    else:
        entities = data.get("entities", [])

    if not entities:
        return JSONResponse(
            status_code=400,
            content={"error": "No entities selected for removal."},
        )

    payload, removed = _remove_entities_from_cache(cache_id, entities)
    if payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    removed_highlights = [
        _compact_render_highlight(
            geometry_to_render_pct(f["geometry"], payload["bounds"])
        )
        for f in removed[:MAX_EXTRACT_HIGHLIGHTS_RETURN]
    ]
    return {
        "cache_id": cache_id,
        "removed_count": len(removed),
        "removed_highlights": removed_highlights,
        "removed_highlight_count_total": len(removed),
        "remaining_entity_count": len(payload["fingerprints"]),
    }


@app.post("/download_selected_dxf")
async def download_selected_dxf(data: dict = Body(...)):
    cache_id = data.get("cache_id")
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    template_id = data.get("template_id")
    if template_id:
        stored_template = _load_extracted_template(cache_id, template_id)
        if stored_template is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found. Please extract template again."},
            )
        entities = stored_template.get("entities", [])
    else:
        entities = data.get("entities", [])

    if not entities:
        return JSONResponse(
            status_code=400,
            content={"error": "No selected entities available for DXF export."},
        )

    selected_features = _match_selected_features(cache_payload, entities)
    if not selected_features:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Selected entities are no longer available in the current cache."
            },
        )

    doc = _build_selected_dxf(selected_features)
    tmp = tempfile.NamedTemporaryFile(suffix=".dxf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    doc.saveas(tmp_path)
    filename = f"selected_entities_{cache_id[:8]}.dxf"
    return FileResponse(
        tmp_path,
        media_type="application/dxf",
        filename=filename,
        background=BackgroundTask(
            lambda: os.path.exists(tmp_path) and os.remove(tmp_path)
        ),
    )


@app.post("/download_matched_dxf")
async def download_matched_dxf(data: dict = Body(...)):
    cache_id = data.get("cache_id")
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    matches = data.get("matches", [])
    if not matches:
        return JSONResponse(
            status_code=400,
            content={"error": "No matches available for DXF export."},
        )

    doc, exported = _build_matches_dxf(matches, cache_payload["bounds"])
    if exported == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Matched results do not contain downloadable highlight geometry."
            },
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".dxf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    doc.saveas(tmp_path)
    filename = f"matched_entities_{cache_id[:8]}.dxf"
    return FileResponse(
        tmp_path,
        media_type="application/dxf",
        filename=filename,
        background=BackgroundTask(
            lambda: os.path.exists(tmp_path) and os.remove(tmp_path)
        ),
    )


def _build_match(
    anchor_idx_or_feat,
    anchor_tmpl,
    group_center,
    used,
    bounds,
    all_fp,
    *,
    max_highlights=MAX_SCAN_HIGHLIGHTS_RETURN,
    match_score=None,
):
    """Shared match result builder. Anchor can be int(index) or dict(feature)."""
    if isinstance(anchor_idx_or_feat, (int, np.integer)):
        af = all_fp[int(anchor_idx_or_feat)]
    else:
        af = anchor_idx_or_feat
    cx, cy = af["x"], af["y"]
    dx = cx - anchor_tmpl["x"]
    dy = cy - anchor_tmpl["y"]

    if group_center:
        mcx = group_center["x"] + dx
        mcy = group_center["y"] + dy
    else:
        mcx, mcy = cx, cy

    rpx = (mcx - bounds["min_x"]) / bounds["width"]
    rpy = (bounds["max_y"] - mcy) / bounds["height"]

    mf = [af] + [all_fp[i] for i in used]
    selected = mf if max_highlights is None else mf[:max_highlights]
    hl = [
        _compact_render_highlight(geometry_to_render_pct(f["geometry"], bounds))
        for f in selected
    ]

    return {
        "dxf_x": round(mcx, 3),
        "dxf_y": round(mcy, 3),
        "render_pct_x": round(rpx, 4),
        "render_pct_y": round(rpy, 4),
        "match_score": (
            round(float(match_score), 6) if match_score is not None else None
        ),
        "highlights": hl,
        "highlight_count_total": len(mf),
        "highlight_count_returned": len(hl),
        "highlight_truncated": len(hl) < len(mf),
    }


def _trim_match_highlights(
    matches, max_with_highlights=MAX_SCAN_MATCHES_WITH_HIGHLIGHTS
):
    """Keep highlights only for the first N matches to cap payload size."""
    if len(matches) <= max_with_highlights:
        return matches
    for m in matches[max_with_highlights:]:
        m["highlights"] = []
        m["highlight_count_returned"] = 0
        m["highlight_truncated"] = m.get("highlight_count_total", 0) > 0
    return matches


def _dedupe_and_sort_matches(matches):
    """Deduplicate by rendered center; keep the lowest score, then sort by score."""
    best = {}
    for m in matches:
        k = (m["render_pct_x"], m["render_pct_y"])
        prev = best.get(k)
        ms = m.get("match_score")
        ps = None if prev is None else prev.get("match_score")
        if prev is None or (ms is not None and (ps is None or ms < ps)):
            best[k] = m
    out = list(best.values())
    out.sort(key=lambda m: (m.get("match_score") is None, m.get("match_score", 1e9)))
    return out


def _finalize_scan_stats(stats, total_start):
    stats["elapsed_ms"] = int((time.perf_counter() - total_start) * 1000)
    scanned = stats.get("neighbor_sets_scanned", 0)
    total_neighbors = stats.get("neighbor_features_total", 0)
    stats["avg_neighbors_per_anchor"] = round(
        (total_neighbors / scanned) if scanned else 0.0, 2
    )
    return stats


@app.post("/scan")
async def scan_dxf(template: dict = Body(...)):
    """Standard scan: cache + KD-Tree + bipartite matching (Python loop)."""
    total_start = time.perf_counter()
    stats = {
        "engine": "standard",
        "template_entity_count": 0,
        "single_entity_mode": False,
        "score_max": None,
        "score_reject_count": 0,
        "candidate_anchor_base": 0,
        "candidate_anchor_after_plugin": 0,
        "neighbor_sets_scanned": 0,
        "neighbor_features_total": 0,
        "adjacency_fail_count": 0,
        "matching_fail_count": 0,
        "raw_match_count": 0,
        "unique_match_count": 0,
        "prefilter_ms": 0,
        "matching_ms": 0,
    }
    runtime_config = _coerce_runtime_config(template.get("settings"))
    enabled_plugins = _resolve_enabled_plugins(template.get("plugins"))
    scan_highlights_limit = int(runtime_config["scan_highlights_return_limit"])
    match_highlight_payload_limit = int(
        runtime_config["scan_matches_with_highlights_limit"]
    )
    stats["plugins_enabled"] = sorted(enabled_plugins)
    stats["runtime_config"] = runtime_config
    cache_payload = _get_cache(template.get("cache_id"))
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    bounds = cache_payload["bounds"]
    all_fp = cache_payload["fingerprints"]
    tree = cache_payload["tree"]
    type_index = cache_payload["type_index"]
    sizes = cache_payload["sizes"]

    template_id = template.get("template_id")
    if template_id:
        stored_template = _load_extracted_template(
            template.get("cache_id"), template_id
        )
        if stored_template is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found. Please extract template again."},
            )
        entities = stored_template.get("entities", [])
        group_center = stored_template.get("group_center", None)
    else:
        entities = template.get("entities", [])
        group_center = template.get("group_center", None)

    score_max = _resolve_score_max(template, runtime_config)
    stats["score_max"] = score_max
    stats["template_entity_count"] = len(entities)
    if not entities or tree is None:
        return {
            "match_count": 0,
            "matches": [],
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    # Single-entity template: search directly by type + size
    if len(entities) == 1:
        stats["single_entity_mode"] = True
        anchor_tmpl = entities[0]
        anchor_size_tol = max(
            runtime_config["size_tol_min"],
            anchor_tmpl["size"] * runtime_config["size_tol_ratio"],
        )
        stats["candidate_anchor_base"] = sum(
            1
            for f in all_fp
            if f["type"] == anchor_tmpl["type"]
            and abs(f["size"] - anchor_tmpl["size"]) < anchor_size_tol
        )
        t_prefilter = time.perf_counter()
        matched_indices = _single_entity_matches(
            all_fp,
            anchor_tmpl,
            config=runtime_config,
            enabled_plugins=enabled_plugins,
        )
        stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)
        stats["candidate_anchor_after_plugin"] = len(matched_indices)
        t_matching = time.perf_counter()
        matches_found = []
        for i in matched_indices:
            f = all_fp[i]
            s = _pair_match_score(
                anchor_tmpl,
                f,
                size_tol=anchor_size_tol,
                config=runtime_config,
            )
            if s > score_max:
                stats["score_reject_count"] += 1
                continue
            matches_found.append(
                _build_match(
                    i,
                    anchor_tmpl,
                    group_center,
                    set(),
                    bounds,
                    all_fp,
                    max_highlights=scan_highlights_limit,
                    match_score=s,
                )
            )
        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
        stats["raw_match_count"] = len(matches_found)
        unique = _dedupe_and_sort_matches(matches_found)
        stats["unique_match_count"] = len(unique)
        return {
            "match_count": len(unique),
            "matches": _trim_match_highlights(
                unique,
                max_with_highlights=match_highlight_payload_limit,
            ),
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    # Pick the rarest entity as anchor
    t_prefilter = time.perf_counter()
    anchor_idx = _pick_anchor(
        entities,
        all_fp,
        type_index,
        sizes,
        config=runtime_config,
        enabled_plugins=enabled_plugins,
    )
    anchor_tmpl = entities[anchor_idx]
    others = [entities[i] for i in range(len(entities)) if i != anchor_idx]

    anchor_pt = (anchor_tmpl["x"], anchor_tmpl["y"])
    anchor_size_tol = max(
        runtime_config["size_tol_min"],
        anchor_tmpl["size"] * runtime_config["size_tol_ratio"],
    )

    targets = []
    for o in others:
        d = calculate_distance(anchor_pt, (o["x"], o["y"]))
        targets.append(
            {
                "entity": o,
                "type": o["type"],
                "size": o["size"],
                "size_tol": max(
                    runtime_config["size_tol_min"],
                    o["size"] * runtime_config["size_tol_ratio"],
                ),
                "dist": d,
                "dist_tol": max(
                    runtime_config["dist_tol_min"],
                    d * runtime_config["dist_tol_ratio"],
                ),
            }
        )

    max_tol = max(t["dist_tol"] for t in targets)
    search_r = max(t["dist"] for t in targets) + max_tol

    potential_base = [
        (i, f)
        for i, f in enumerate(all_fp)
        if f["type"] == anchor_tmpl["type"]
        and abs(f["size"] - anchor_tmpl["size"]) < anchor_size_tol
    ]
    stats["candidate_anchor_base"] = len(potential_base)
    potential = [
        (i, f)
        for i, f in potential_base
        if _single_entity_plugin_pass(anchor_tmpl, f, runtime_config, enabled_plugins)
    ]
    stats["candidate_anchor_after_plugin"] = len(potential)
    stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)

    matches_found = []
    t_matching = time.perf_counter()
    for ac_idx, ac in potential:
        cp = (ac["x"], ac["y"])
        local_set = set(tree.query_ball_point(cp, search_r)) - {ac_idx}
        stats["neighbor_sets_scanned"] += 1
        stats["neighbor_features_total"] += len(local_set)

        # Build scored adjacency list
        adj_scored = []
        skip = False
        for t in targets:
            valid = []
            for idx in local_set:
                f = all_fp[idx]
                if not _entity_basic_match(
                    t["entity"],
                    f,
                    size_tol=t["size_tol"],
                    config=runtime_config,
                    enabled_plugins=enabled_plugins,
                ):
                    continue
                d = calculate_distance(cp, (f["x"], f["y"]))
                dist_err = abs(d - t["dist"])
                if dist_err < t["dist_tol"]:
                    edge_score = _pair_match_score(
                        t["entity"],
                        f,
                        size_tol=t["size_tol"],
                        config=runtime_config,
                        dist_error_ratio=(dist_err / t["dist_tol"]),
                    )
                    valid.append((idx, edge_score))
            if not valid:
                skip = True
                break
            adj_scored.append(valid)

        if skip:
            stats["adjacency_fail_count"] += 1
            continue

        matched = _best_scored_matching(adj_scored)
        if matched is not None:
            matched_set, edge_mean_score = matched
            anchor_score = _pair_match_score(
                anchor_tmpl,
                ac,
                size_tol=anchor_size_tol,
                config=runtime_config,
            )
            total_score = (
                runtime_config["match_score_anchor_weight"] * anchor_score
                + (1.0 - runtime_config["match_score_anchor_weight"]) * edge_mean_score
            )
            if total_score > score_max:
                stats["score_reject_count"] += 1
                continue
            matches_found.append(
                _build_match(
                    ac_idx,
                    anchor_tmpl,
                    group_center,
                    matched_set,
                    bounds,
                    all_fp,
                    max_highlights=scan_highlights_limit,
                    match_score=total_score,
                )
            )
        else:
            stats["matching_fail_count"] += 1

    stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
    stats["raw_match_count"] = len(matches_found)
    unique = _dedupe_and_sort_matches(matches_found)
    stats["unique_match_count"] = len(unique)
    return {
        "match_count": len(unique),
        "matches": _trim_match_highlights(
            unique,
            max_with_highlights=match_highlight_payload_limit,
        ),
        "scan_stats": _finalize_scan_stats(stats, total_start),
    }


@app.post("/scan_fast")
async def scan_dxf_fast(template: dict = Body(...)):
    """Fast scan: cache + batch KD-Tree + numpy vectorization + bipartite matching."""
    total_start = time.perf_counter()
    stats = {
        "engine": "fast",
        "template_entity_count": 0,
        "single_entity_mode": False,
        "score_max": None,
        "score_reject_count": 0,
        "candidate_anchor_base": 0,
        "candidate_anchor_after_plugin": 0,
        "neighbor_sets_scanned": 0,
        "neighbor_features_total": 0,
        "adjacency_fail_count": 0,
        "matching_fail_count": 0,
        "raw_match_count": 0,
        "unique_match_count": 0,
        "prefilter_ms": 0,
        "matching_ms": 0,
    }
    runtime_config = _coerce_runtime_config(template.get("settings"))
    enabled_plugins = _resolve_enabled_plugins(template.get("plugins"))
    scan_highlights_limit = int(runtime_config["scan_highlights_return_limit"])
    match_highlight_payload_limit = int(
        runtime_config["scan_matches_with_highlights_limit"]
    )
    stats["plugins_enabled"] = sorted(enabled_plugins)
    stats["runtime_config"] = runtime_config
    cache_payload = _get_cache(template.get("cache_id"))
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    bounds = cache_payload["bounds"]
    all_fp = cache_payload["fingerprints"]
    coords = cache_payload["coords"]
    sizes = cache_payload["sizes"]
    tree = cache_payload["tree"]
    type_index = cache_payload["type_index"]

    template_id = template.get("template_id")
    if template_id:
        stored_template = _load_extracted_template(
            template.get("cache_id"), template_id
        )
        if stored_template is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found. Please extract template again."},
            )
        entities = stored_template.get("entities", [])
        group_center = stored_template.get("group_center", None)
    else:
        entities = template.get("entities", [])
        group_center = template.get("group_center", None)

    score_max = _resolve_score_max(template, runtime_config)
    stats["score_max"] = score_max
    stats["template_entity_count"] = len(entities)
    if not entities or tree is None:
        return {
            "match_count": 0,
            "matches": [],
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    # Single-entity template: search directly by type + size
    if len(entities) == 1:
        stats["single_entity_mode"] = True
        anchor_tmpl = entities[0]
        anchor_size_tol = max(
            runtime_config["size_tol_min"],
            anchor_tmpl["size"] * runtime_config["size_tol_ratio"],
        )
        stats["candidate_anchor_base"] = sum(
            1
            for f in all_fp
            if f["type"] == anchor_tmpl["type"]
            and abs(f["size"] - anchor_tmpl["size"]) < anchor_size_tol
        )
        t_prefilter = time.perf_counter()
        matched_indices = _single_entity_matches(
            all_fp,
            anchor_tmpl,
            config=runtime_config,
            enabled_plugins=enabled_plugins,
        )
        stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)
        stats["candidate_anchor_after_plugin"] = len(matched_indices)

        t_matching = time.perf_counter()
        matches_found = []
        for ai in matched_indices:
            f = all_fp[int(ai)]
            s = _pair_match_score(
                anchor_tmpl,
                f,
                size_tol=anchor_size_tol,
                config=runtime_config,
            )
            if s > score_max:
                stats["score_reject_count"] += 1
                continue
            matches_found.append(
                _build_match(
                    int(ai),
                    anchor_tmpl,
                    group_center,
                    set(),
                    bounds,
                    all_fp,
                    max_highlights=scan_highlights_limit,
                    match_score=s,
                )
            )
        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
        stats["raw_match_count"] = len(matches_found)
        unique = _dedupe_and_sort_matches(matches_found)
        stats["unique_match_count"] = len(unique)
        return {
            "match_count": len(unique),
            "matches": _trim_match_highlights(
                unique,
                max_with_highlights=match_highlight_payload_limit,
            ),
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    # Pick the rarest entity as anchor
    t_prefilter = time.perf_counter()
    anchor_idx = _pick_anchor(
        entities,
        all_fp,
        type_index,
        sizes,
        config=runtime_config,
        enabled_plugins=enabled_plugins,
    )
    anchor_tmpl = entities[anchor_idx]
    others = [entities[i] for i in range(len(entities)) if i != anchor_idx]

    anchor_pt = np.array([anchor_tmpl["x"], anchor_tmpl["y"]])
    anchor_size_tol = max(
        runtime_config["size_tol_min"],
        anchor_tmpl["size"] * runtime_config["size_tol_ratio"],
    )

    targets = []
    for o in others:
        d = float(np.linalg.norm(anchor_pt - [o["x"], o["y"]]))
        targets.append(
            {
                "entity": o,
                "type": o["type"],
                "size": o["size"],
                "size_tol": max(
                    runtime_config["size_tol_min"],
                    o["size"] * runtime_config["size_tol_ratio"],
                ),
                "dist": d,
                "dist_tol": max(
                    runtime_config["dist_tol_min"],
                    d * runtime_config["dist_tol_ratio"],
                ),
            }
        )

    max_tol = max(t["dist_tol"] for t in targets)
    search_r = max(t["dist"] for t in targets) + max_tol

    # Numpy pre-filter for anchor candidates
    a_type = anchor_tmpl["type"]
    if a_type not in type_index:
        stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)
        return {
            "match_count": 0,
            "matches": [],
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }
    a_idx = type_index[a_type]
    mask = np.abs(sizes[a_idx] - anchor_tmpl["size"]) < anchor_size_tol
    potential_base = a_idx[mask]
    stats["candidate_anchor_base"] = int(len(potential_base))
    potential = potential_base
    potential = np.array(
        [
            int(i)
            for i in potential
            if _single_entity_plugin_pass(
                anchor_tmpl,
                all_fp[int(i)],
                runtime_config,
                enabled_plugins,
            )
        ],
        dtype=np.intp,
    )
    stats["candidate_anchor_after_plugin"] = int(len(potential))
    stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)

    if len(potential) == 0:
        return {
            "match_count": 0,
            "matches": [],
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    # Batch KD-Tree query: fetch neighbors for all anchors in one call
    all_locals = tree.query_ball_point(coords[potential], search_r)

    # Prebuild type index arrays
    target_types = list(set(t["type"] for t in targets))
    type_idx_arrays = {t: type_index[t] for t in target_types if t in type_index}

    matches_found = []
    t_matching = time.perf_counter()

    for pi, ai in enumerate(potential):
        cand = coords[ai]
        local_arr = np.array(all_locals[pi], dtype=np.intp)
        if len(local_arr) == 0:
            continue

        # Exclude anchor itself
        local_arr = local_arr[local_arr != ai]
        stats["neighbor_sets_scanned"] += 1
        stats["neighbor_features_total"] += int(len(local_arr))

        # Pre-intersection: local intersection type_index
        local_by_type = {}
        for tt in target_types:
            if tt in type_idx_arrays:
                local_by_type[tt] = np.intersect1d(local_arr, type_idx_arrays[tt])
            else:
                local_by_type[tt] = np.array([], dtype=np.intp)

        # Build scored adjacency list (vectorized size + distance filtering)
        adj_scored = []
        skip = False
        for t in targets:
            cands = local_by_type.get(t["type"], np.array([], dtype=np.intp))
            if len(cands) == 0:
                skip = True
                break
            s_mask = np.abs(sizes[cands] - t["size"]) < t["size_tol"]
            cands = cands[s_mask]
            if len(cands) == 0:
                skip = True
                break
            cands = np.array(
                [
                    int(ci)
                    for ci in cands
                    if _single_entity_plugin_pass(
                        t["entity"],
                        all_fp[int(ci)],
                        runtime_config,
                        enabled_plugins,
                    )
                ],
                dtype=np.intp,
            )
            if len(cands) == 0:
                skip = True
                break
            dists = np.linalg.norm(coords[cands] - cand, axis=1)
            dist_err = np.abs(dists - t["dist"])
            d_mask = dist_err < t["dist_tol"]
            valid = cands[d_mask]
            if len(valid) == 0:
                skip = True
                break
            valid_dist_err = dist_err[d_mask]
            scored = []
            for vi, de in zip(valid.tolist(), valid_dist_err.tolist()):
                edge_score = _pair_match_score(
                    t["entity"],
                    all_fp[int(vi)],
                    size_tol=t["size_tol"],
                    config=runtime_config,
                    dist_error_ratio=(de / t["dist_tol"]),
                )
                scored.append((int(vi), edge_score))
            adj_scored.append(scored)

        if skip:
            stats["adjacency_fail_count"] += 1
            continue

        matched = _best_scored_matching(adj_scored)
        if matched is not None:
            matched_set, edge_mean_score = matched
            anchor_score = _pair_match_score(
                anchor_tmpl,
                all_fp[int(ai)],
                size_tol=anchor_size_tol,
                config=runtime_config,
            )
            total_score = (
                runtime_config["match_score_anchor_weight"] * anchor_score
                + (1.0 - runtime_config["match_score_anchor_weight"]) * edge_mean_score
            )
            if total_score > score_max:
                stats["score_reject_count"] += 1
                continue
            matches_found.append(
                _build_match(
                    int(ai),
                    anchor_tmpl,
                    group_center,
                    matched_set,
                    bounds,
                    all_fp,
                    max_highlights=scan_highlights_limit,
                    match_score=total_score,
                )
            )
        else:
            stats["matching_fail_count"] += 1

    stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
    stats["raw_match_count"] = len(matches_found)
    unique = _dedupe_and_sort_matches(matches_found)
    stats["unique_match_count"] = len(unique)
    return {
        "match_count": len(unique),
        "matches": _trim_match_highlights(
            unique,
            max_with_highlights=match_highlight_payload_limit,
        ),
        "scan_stats": _finalize_scan_stats(stats, total_start),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
