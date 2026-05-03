from collections import Counter
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
import socket
from typing import Any, Callable, Optional
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, UploadFile, File, Body, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
import ezdxf
from ezdxf.bbox import extents
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.math import Matrix44
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon

# Initialize FastAPI app
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "templates" / "index.html"
TEMPLATE_LIBRARY_PATH = BASE_DIR / "data" / "template_library.json"
TEMPLATE_LIBRARY_DB_PATH = BASE_DIR / "data" / "template_library.sqlite"
NO_STORE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}

# --- Algorithm constants ---
SNAP_DECIMALS = 6  # Coordinate rounding precision for stable geometry fingerprints
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
BOUNDS_EPS = 1e-9
MAX_INSERT_EXPLODE_DEPTH = 8
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
]


def snap(val):
    """Round coordinates to fixed precision for stable geometry fingerprints."""
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


class AgentViewProposal(BaseModel):
    name: str
    roi_pct: Optional[list[list[float]]] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class AgentSeedCandidate(BaseModel):
    target_class: str = "SMD"
    view_name: Optional[str] = None
    label: Optional[str] = None
    template_id: Optional[str] = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    group_center: Optional[dict[str, float]] = None
    click_pct: Optional[list[float]] = None
    polygon_pct: Optional[list[list[float]]] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class AgentProviderConfig(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    llm_model: Optional[str] = None
    vlm_model: Optional[str] = None
    timeout_seconds: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    send_image: bool = False


class AgentRunRequest(BaseModel):
    cache_id: str
    instruction: str
    target_classes: list[str] = Field(default_factory=lambda: ["SMD"])
    target_descriptions: dict[str, str] = Field(default_factory=dict)
    views: list[AgentViewProposal] = Field(default_factory=list)
    seed_candidates: list[AgentSeedCandidate] = Field(default_factory=list)
    plugins: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class AgentProposeRequest(BaseModel):
    cache_id: str
    instruction: str
    target_classes: list[str] = Field(default_factory=lambda: ["SMD"])
    target_descriptions: dict[str, str] = Field(default_factory=dict)
    views: list[AgentViewProposal] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    provider: Optional[AgentProviderConfig] = None
    render_image_data_url: Optional[str] = None


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


def _matrix_from_flat(raw):
    if not raw:
        return None
    try:
        return Matrix44([float(v) for v in raw])
    except Exception:
        return None


def _dxf_point_to_render_pct(point, bounds, render_mapping=None):
    mapping = render_mapping or {}
    matrix = _matrix_from_flat(mapping.get("matrix"))
    viewbox_x = mapping.get("viewbox_x")
    viewbox_y = mapping.get("viewbox_y")
    viewbox_w = mapping.get("viewbox_w")
    viewbox_h = mapping.get("viewbox_h")
    if matrix and viewbox_w and viewbox_h:
        try:
            p = matrix.transform((float(point[0]), float(point[1]), 0.0))
            return [
                (float(p[0]) - float(viewbox_x or 0.0)) / float(viewbox_w),
                (float(p[1]) - float(viewbox_y or 0.0)) / float(viewbox_h),
            ]
        except Exception:
            pass
    return [
        (point[0] - bounds["min_x"]) / bounds["width"],
        (bounds["max_y"] - point[1]) / bounds["height"],
    ]


def geometry_to_render_pct(geom, bounds, render_mapping=None):
    """Convert DXF geometry coordinates to SVG render percentage coordinates."""
    w, h = bounds["width"], bounds["height"]
    min_x, max_y = bounds["min_x"], bounds["max_y"]
    if geom["kind"] == "circle":
        center_pct = _dxf_point_to_render_pct(
            (geom["cx"], geom["cy"]), bounds, render_mapping
        )
        edge_pct = _dxf_point_to_render_pct(
            (geom["cx"] + geom["r"], geom["cy"]), bounds, render_mapping
        )
        return {
            "kind": "circle",
            "cx_pct": center_pct[0],
            "cy_pct": center_pct[1],
            "r_pct": abs(edge_pct[0] - center_pct[0]) or geom["r"] / w,
        }
    else:  # polyline
        return {
            "kind": "polyline",
            "points_pct": [
                _dxf_point_to_render_pct((px, py), bounds, render_mapping)
                for px, py in geom["points"]
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


def _make_bounds(min_x, min_y, max_x, max_y):
    width = max(float(max_x) - float(min_x), BOUNDS_EPS)
    height = max(float(max_y) - float(min_y), BOUNDS_EPS)
    return {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "width": width,
        "height": height,
    }


def get_dxf_bounds(msp):
    """Get DXF extents and dimensions."""
    ext = extents(msp)
    return _make_bounds(ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y)


def _render_pct_to_dxf_point(pt, bounds, render_mapping=None):
    mapping = render_mapping or {}
    inv = _matrix_from_flat(mapping.get("inv_matrix"))
    viewbox_x = mapping.get("viewbox_x")
    viewbox_y = mapping.get("viewbox_y")
    viewbox_w = mapping.get("viewbox_w")
    viewbox_h = mapping.get("viewbox_h")
    if inv and viewbox_w and viewbox_h:
        try:
            x_vb = float(viewbox_x or 0.0) + float(pt[0]) * float(viewbox_w)
            y_vb = float(viewbox_y or 0.0) + float(pt[1]) * float(viewbox_h)
            p = inv.transform((x_vb, y_vb, 0.0))
            return (float(p[0]), float(p[1]))
        except Exception:
            pass
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
    render_mapping = cache_payload.get("render_mapping")
    dxf_vertices = [
        _render_pct_to_dxf_point(pt, bounds, render_mapping) for pt in polygon_pct
    ]
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
    render_mapping = cache_payload.get("render_mapping")
    pick_point = Point(*_render_pct_to_dxf_point(click_pct, bounds, render_mapping))
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
    cache_id,
    bounds,
    entities_found,
    selector_mode,
    runtime_config=None,
    render_mapping=None,
):
    config = runtime_config or DEFAULT_RUNTIME_CONFIG
    highlight_limit = int(config["extract_highlights_return_limit"])
    preview_limit = int(config["extract_entities_preview_limit"])
    if not entities_found:
        return {
            "selector_mode": selector_mode,
            "entities": [],
            "highlights": [],
            "highlight_labels": [],
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
        geometry_to_render_pct(e["geometry"], bounds, render_mapping)
        for e in entities_found
    ]
    highlights = [
        _compact_render_highlight(h) for h in all_highlights[:highlight_limit]
    ]
    highlight_labels = [
        {
            "handleIDs": entity.get("handleIDs", []),
            "x": entity.get("x"),
            "y": entity.get("y"),
            "type": entity.get("type"),
        }
        for entity in entities_clean[:highlight_limit]
    ]

    return {
        "cache_id": cache_id,
        "template_id": template_id,
        "selector_mode": selector_mode,
        "group_center": template_payload["group_center"],
        "entity_count": len(entities_clean),
        "entities_preview": entities_clean[:preview_limit],
        "highlights": highlights,
        "highlight_labels": highlight_labels,
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


def _entity_handle_id(entity):
    handle = getattr(entity.dxf, "handle", None)
    return str(handle) if handle is not None else None


def _unique_handle_ids(handle_ids):
    seen = set()
    unique = []
    for handle_id in handle_ids:
        if not handle_id or handle_id in seen:
            continue
        seen.add(handle_id)
        unique.append(handle_id)
    return unique


def _composite_feature_from_entity(geom, handle_id):
    centroid = geom.centroid
    return {
        "type": "COMPOSITE_SHAPE",
        "size": round(geom.length, 3),
        "x": round(centroid.x, 3),
        "y": round(centroid.y, 3),
        "shape_sig": _polyline_shape_signature(list(geom.coords)),
        "geometry": {
            "kind": "polyline",
            "points": [list(c) for c in geom.coords],
        },
        "handleIDs": [handle_id] if handle_id else [],
    }


def _bbox_feature_from_entity(entity, handle_id):
    try:
        box = extents([entity])
        if not box.has_data:
            return None
        min_x, min_y = box.extmin.x, box.extmin.y
        max_x, max_y = box.extmax.x, box.extmax.y
    except Exception:
        return None
    width = max_x - min_x
    height = max_y - min_y
    if abs(width) <= BOUNDS_EPS and abs(height) <= BOUNDS_EPS:
        return None
    points = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
    ]
    return {
        "type": entity.dxftype(),
        "size": round(math.hypot(width, height), 3),
        "x": round((min_x + max_x) / 2, 3),
        "y": round((min_y + max_y) / 2, 3),
        "geometry": {
            "kind": "polyline",
            "points": points,
        },
        "handleIDs": [handle_id] if handle_id else [],
    }


def _iter_flat_entities(entities, *, depth=0, max_depth=MAX_INSERT_EXPLODE_DEPTH):
    """Yield drawable entities, expanding INSERT block references recursively."""
    if depth > max_depth:
        return
    for entity in entities:
        handle_id = _entity_handle_id(entity)
        if entity.dxftype() == "INSERT":
            if depth >= max_depth:
                continue
            try:
                for child, child_handle_id in _iter_flat_entities(
                    entity.virtual_entities(),
                    depth=depth + 1,
                    max_depth=max_depth,
                ):
                    yield child, child_handle_id or handle_id
            except Exception:
                continue
            continue
        yield entity, handle_id


def extract_template_features(
    msp, roi_box=None, *, flatten_tol=DEFAULT_FLATTEN_TOL, skip_ellipse_spline=False
):
    """
    Smart feature extraction v2:
    1. Preserve one fingerprint per source DXF entity.
    2. Supports CIRCLE, LINE, LWPOLYLINE, ARC, POLYLINE, ELLIPSE, SPLINE.
    3. Snap sampled coordinates for stable shape signatures.
    4. Apply ROI filtering last (centroid must be inside ROI).
    """
    features = []

    for entity, handle_id in _iter_flat_entities(msp):
        etype = entity.dxftype()

        if etype == "CIRCLE":
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            features.append(
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
                    "handleIDs": [handle_id] if handle_id else [],
                }
            )

        elif etype == "LINE":
            p1, p2 = entity.dxf.start, entity.dxf.end
            geom = LineString([(snap(p1.x), snap(p1.y)), (snap(p2.x), snap(p2.y))])
            features.append(_composite_feature_from_entity(geom, handle_id))

        elif etype == "LWPOLYLINE":
            pts = [(snap(p[0]), snap(p[1])) for p in entity.get_points()]
            if entity.is_closed and len(pts) >= 3:
                pts.append(pts[0])
            if len(pts) >= 2:
                geom = LineString(pts)
                features.append(_composite_feature_from_entity(geom, handle_id))

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
            geom = LineString(pts)
            features.append(_composite_feature_from_entity(geom, handle_id))

        elif etype == "POLYLINE":
            try:
                pts = [
                    (snap(v.dxf.location.x), snap(v.dxf.location.y))
                    for v in entity.vertices
                ]
                if entity.is_closed and len(pts) >= 3:
                    pts.append(pts[0])
                if len(pts) >= 2:
                    geom = LineString(pts)
                    features.append(_composite_feature_from_entity(geom, handle_id))
            except Exception:
                pass

        elif etype in ("ELLIPSE", "SPLINE"):
            if skip_ellipse_spline:
                continue
            try:
                pts = [(snap(p.x), snap(p.y)) for p in entity.flattening(flatten_tol)]
                if len(pts) >= 2:
                    geom = LineString(pts)
                    features.append(_composite_feature_from_entity(geom, handle_id))
            except Exception:
                pass

        else:
            fallback = _bbox_feature_from_entity(entity, handle_id)
            if fallback is not None:
                features.append(fallback)

    # --- ROI filtering: keep features whose centroids are inside ROI ---
    if roi_box is not None:
        features = [f for f in features if roi_box.covers(Point(f["x"], f["y"]))]

    return features


# --- Session cache store ---
_cache_store = {}
_cache_lock = threading.Lock()
_template_library_lock = threading.Lock()


def _build_cache_payload(msp, *, fast_build=False, render_mapping=None):
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
        "render_mapping": render_mapping,
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


def _normalize_template_category(raw_category):
    text = str(raw_category or "").strip()
    return text or "Uncategorized"


def _default_template_name():
    return "Template " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _sanitize_template_group_center(group_center):
    if not isinstance(group_center, dict):
        return None
    try:
        x = float(group_center.get("x"))
        y = float(group_center.get("y"))
    except (TypeError, ValueError):
        return None
    return {"x": round(x, 6), "y": round(y, 6)}


def _compute_group_center_from_entities(entities):
    if not entities:
        return None
    return {
        "x": round(sum(float(e["x"]) for e in entities) / len(entities), 6),
        "y": round(sum(float(e["y"]) for e in entities) / len(entities), 6),
    }


def _sanitize_template_entities(entities):
    if not isinstance(entities, list):
        return []
    cleaned = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if not all(key in entity for key in ("type", "size", "x", "y")):
            continue
        try:
            item = {
                "type": str(entity["type"]),
                "size": round(float(entity["size"]), 6),
                "x": round(float(entity["x"]), 6),
                "y": round(float(entity["y"]), 6),
            }
        except (TypeError, ValueError):
            continue
        shape_sig = entity.get("shape_sig")
        if isinstance(shape_sig, dict):
            item["shape_sig"] = {
                key: shape_sig[key]
                for key in ("point_count", "bbox_diag", "aspect_ratio")
                if key in shape_sig
            }
        cleaned.append(item)
    return cleaned


def _default_template_library_version_label(timestamp=None):
    ts = float(timestamp or time.time())
    return "Version " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _normalize_saved_template_record(record):
    if not isinstance(record, dict) or not record.get("id"):
        return None
    entities = _sanitize_template_entities(record.get("entities"))
    if not entities:
        return None
    group_center = _sanitize_template_group_center(record.get("group_center"))
    if group_center is None:
        group_center = _compute_group_center_from_entities(entities)
    created_at = float(record.get("created_at") or 0.0)
    updated_at = float(record.get("updated_at") or created_at)
    return {
        "id": str(record["id"]),
        "name": str(record.get("name") or _default_template_name()).strip()
        or _default_template_name(),
        "category": _normalize_template_category(record.get("category")),
        "group_center": group_center,
        "entities": entities,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _normalize_template_library_version(raw_version, fallback_label=None):
    now = time.time()
    if not isinstance(raw_version, dict):
        raw_version = {}
    created_at = float(raw_version.get("created_at") or now)
    updated_at = float(raw_version.get("updated_at") or created_at)
    label = str(
        raw_version.get("label")
        or raw_version.get("version_label")
        or raw_version.get("name")
        or fallback_label
        or _default_template_library_version_label(created_at)
    ).strip() or _default_template_library_version_label(created_at)
    templates = []
    for record in raw_version.get("templates", []):
        normalized = _normalize_saved_template_record(record)
        if normalized is not None:
            templates.append(normalized)
    return {
        "id": str(
            raw_version.get("id") or raw_version.get("version_id") or uuid.uuid4().hex
        ),
        "label": label,
        "created_at": created_at,
        "updated_at": updated_at,
        "templates": templates,
    }


def _new_empty_template_library_version(label=None):
    now = time.time()
    return {
        "id": uuid.uuid4().hex,
        "label": str(label or _default_template_library_version_label(now)).strip()
        or _default_template_library_version_label(now),
        "created_at": now,
        "updated_at": now,
        "templates": [],
    }


def _normalize_template_library_payload(raw):
    versions = []
    active_version_id = None

    if isinstance(raw, dict) and isinstance(raw.get("versions"), list):
        active_version_id = str(raw.get("active_version_id") or "").strip() or None
        for idx, version in enumerate(raw.get("versions", [])):
            versions.append(
                _normalize_template_library_version(
                    version,
                    fallback_label="Version " + str(idx + 1),
                )
            )
    else:
        if isinstance(raw, list):
            raw_templates = raw
            raw_version = {"templates": raw_templates}
        elif isinstance(raw, dict):
            raw_version = {
                "id": raw.get("version_id") or raw.get("id"),
                "label": raw.get("version_label") or raw.get("label"),
                "created_at": raw.get("created_at"),
                "updated_at": raw.get("updated_at"),
                "templates": raw.get("templates", []),
            }
        else:
            raw_version = {"templates": []}
        version = _normalize_template_library_version(raw_version)
        versions.append(version)
        active_version_id = version["id"]

    if not versions:
        version = _new_empty_template_library_version()
        versions.append(version)
        active_version_id = version["id"]

    version_ids = {version["id"] for version in versions}
    if active_version_id not in version_ids:
        active_version_id = versions[0]["id"]

    return {
        "active_version_id": active_version_id,
        "versions": versions,
    }


def _read_template_library_locked():
    TEMPLATE_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE_LIBRARY_PATH.exists():
        return _normalize_template_library_payload({"versions": []})
    try:
        raw = json.loads(TEMPLATE_LIBRARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _normalize_template_library_payload({"versions": []})
    return _normalize_template_library_payload(raw)


def _write_template_library_locked(library):
    TEMPLATE_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = TEMPLATE_LIBRARY_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(library, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, TEMPLATE_LIBRARY_PATH)


def _serialize_saved_template(record, *, include_entities=False):
    payload = {
        "id": record["id"],
        "name": record["name"],
        "category": record["category"],
        "group_center": record["group_center"],
        "entity_count": len(record.get("entities", [])),
        "created_at": record.get("created_at", 0.0),
        "updated_at": record.get("updated_at", 0.0),
    }
    if include_entities:
        payload["entities"] = record.get("entities", [])
    return payload


def _serialize_template_library_version(record, *, is_active=False):
    return {
        "id": record["id"],
        "label": record["label"],
        "template_count": len(record.get("templates", [])),
        "created_at": record.get("created_at", 0.0),
        "updated_at": record.get("updated_at", 0.0),
        "is_active": bool(is_active),
    }


def _get_active_template_library_version(library):
    active_version_id = library.get("active_version_id")
    for version in library.get("versions", []):
        if version["id"] == active_version_id:
            return version
    if library.get("versions"):
        library["active_version_id"] = library["versions"][0]["id"]
        return library["versions"][0]
    version = _new_empty_template_library_version()
    library["versions"] = [version]
    library["active_version_id"] = version["id"]
    return version


def _build_template_library_categories(templates):
    categories = {}
    for record in sorted(
        templates,
        key=lambda item: (
            item.get("category", "").lower(),
            item.get("name", "").lower(),
            item.get("created_at", 0.0),
        ),
    ):
        category = record["category"]
        categories.setdefault(category, []).append(_serialize_saved_template(record))
    return [
        {
            "name": name,
            "template_count": len(items),
            "templates": items,
        }
        for name, items in categories.items()
    ]


def _build_template_library_response():
    with _template_library_lock:
        library = _read_template_library_locked()
    active_version = _get_active_template_library_version(library)

    return {
        "active_version_id": active_version["id"],
        "active_version_label": active_version["label"],
        "version_count": len(library["versions"]),
        "versions": [
            _serialize_template_library_version(
                version, is_active=(version["id"] == active_version["id"])
            )
            for version in sorted(
                library["versions"],
                key=lambda item: item.get("created_at", 0.0),
                reverse=True,
            )
        ],
        "template_count": len(active_version["templates"]),
        "categories": _build_template_library_categories(active_version["templates"]),
    }


def _template_library_json_response(payload, *, status_code=200):
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=NO_STORE_HEADERS,
    )


def _save_template_library_entry(name, category, group_center, entities):
    cleaned_entities = _sanitize_template_entities(entities)
    if not cleaned_entities:
        return None
    cleaned_group_center = _sanitize_template_group_center(group_center)
    if cleaned_group_center is None:
        cleaned_group_center = _compute_group_center_from_entities(cleaned_entities)

    now = time.time()
    record = {
        "id": uuid.uuid4().hex,
        "name": str(name or "").strip() or _default_template_name(),
        "category": _normalize_template_category(category),
        "group_center": cleaned_group_center,
        "entities": cleaned_entities,
        "created_at": now,
        "updated_at": now,
    }

    with _template_library_lock:
        library = _read_template_library_locked()
        active_version = _get_active_template_library_version(library)
        active_version["templates"].append(record)
        active_version["updated_at"] = now
        _write_template_library_locked(library)
    return record


def _resolve_template_library_entries(template_ids):
    if not isinstance(template_ids, list):
        return []
    wanted = [str(tid) for tid in template_ids if str(tid).strip()]
    if not wanted:
        return []

    with _template_library_lock:
        library = _read_template_library_locked()

    active_version = _get_active_template_library_version(library)
    records = {record["id"]: record for record in active_version["templates"]}
    return [records[tid] for tid in wanted if tid in records]


def _select_template_library_version(version_id):
    with _template_library_lock:
        library = _read_template_library_locked()
        version_ids = {version["id"] for version in library["versions"]}
        if version_id not in version_ids:
            return None
        library["active_version_id"] = version_id
        _write_template_library_locked(library)
    return library


def _delete_template_library_template(template_id):
    with _template_library_lock:
        library = _read_template_library_locked()
        active_version = _get_active_template_library_version(library)
        kept = [
            template
            for template in active_version.get("templates", [])
            if template.get("id") != template_id
        ]
        if len(kept) == len(active_version.get("templates", [])):
            return None
        active_version["templates"] = kept
        active_version["updated_at"] = time.time()
        _write_template_library_locked(library)
    return library


def _delete_template_library_category(category_name):
    normalized_category = _normalize_template_category(category_name)
    with _template_library_lock:
        library = _read_template_library_locked()
        active_version = _get_active_template_library_version(library)
        current_templates = active_version.get("templates", [])
        kept = [
            template
            for template in current_templates
            if _normalize_template_category(template.get("category"))
            != normalized_category
        ]
        deleted_count = len(current_templates) - len(kept)
        if deleted_count == 0:
            return None, 0
        active_version["templates"] = kept
        active_version["updated_at"] = time.time()
        _write_template_library_locked(library)
    return library, deleted_count


def _delete_template_library_version(version_id):
    with _template_library_lock:
        library = _read_template_library_locked()
        kept = [
            version
            for version in library.get("versions", [])
            if version.get("id") != version_id
        ]
        if len(kept) == len(library.get("versions", [])):
            return None
        if not kept:
            new_version = _new_empty_template_library_version()
            kept = [new_version]
            library["active_version_id"] = new_version["id"]
        elif library.get("active_version_id") == version_id:
            library["active_version_id"] = kept[0]["id"]
        library["versions"] = kept
        _write_template_library_locked(library)
    return library


def _import_template_library_versions(raw):
    imported = _normalize_template_library_payload(raw)
    with _template_library_lock:
        library = _read_template_library_locked()
        active_version = _get_active_template_library_version(library)
        if (
            len(library["versions"]) == 1
            and not active_version.get("templates")
            and active_version["id"] == library.get("active_version_id")
        ):
            library["versions"] = []
            library["active_version_id"] = None

        old_to_new_version_ids = {}
        imported_versions = []
        for version in imported["versions"]:
            cloned_templates = [
                dict(template, entities=list(template.get("entities", [])))
                for template in version.get("templates", [])
            ]
            new_version_id = uuid.uuid4().hex
            old_to_new_version_ids[version["id"]] = new_version_id
            imported_versions.append(
                {
                    "id": new_version_id,
                    "label": version["label"],
                    "created_at": float(version.get("created_at") or time.time()),
                    "updated_at": float(
                        version.get("updated_at")
                        or version.get("created_at")
                        or time.time()
                    ),
                    "templates": cloned_templates,
                }
            )

        library["versions"].extend(imported_versions)
        active_version_id = old_to_new_version_ids.get(imported["active_version_id"])
        if active_version_id:
            library["active_version_id"] = active_version_id
        elif imported_versions:
            library["active_version_id"] = imported_versions[-1]["id"]
        _write_template_library_locked(library)
    return library, imported_versions


def _build_exportable_template_library_version(version):
    return {
        "version_id": version["id"],
        "version_label": version["label"],
        "created_at": version.get("created_at", 0.0),
        "updated_at": version.get("updated_at", 0.0),
        "templates": version.get("templates", []),
    }


def _sqlite_library_connect():
    TEMPLATE_LIBRARY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TEMPLATE_LIBRARY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _sqlite_library_init_locked(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS libraries (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS templates (
            id TEXT PRIMARY KEY,
            library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            group_center_json TEXT NOT NULL,
            entities_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    _sqlite_migrate_legacy_json_locked(conn)
    _sqlite_cleanup_duplicate_empty_libraries_locked(conn)
    _sqlite_ensure_default_library_locked(conn)


def _sqlite_insert_library_locked(conn, library):
    conn.execute(
        """
        INSERT OR REPLACE INTO libraries(id, label, created_at, updated_at)
        VALUES(?, ?, ?, ?)
        """,
        (
            library["id"],
            library["label"],
            float(library.get("created_at") or time.time()),
            float(
                library.get("updated_at") or library.get("created_at") or time.time()
            ),
        ),
    )
    for template in library.get("templates", []):
        conn.execute(
            """
            INSERT OR REPLACE INTO templates(
                id, library_id, name, category, group_center_json, entities_json,
                created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template["id"],
                library["id"],
                template["name"],
                template["category"],
                json.dumps(template["group_center"], ensure_ascii=False),
                json.dumps(template.get("entities", []), ensure_ascii=False),
                float(template.get("created_at") or time.time()),
                float(
                    template.get("updated_at")
                    or template.get("created_at")
                    or time.time()
                ),
            ),
        )


def _sqlite_migrate_legacy_json_locked(conn):
    migrated = conn.execute(
        "SELECT value FROM app_meta WHERE key = 'legacy_json_migrated'"
    ).fetchone()
    if migrated is not None:
        return
    has_libraries = conn.execute("SELECT COUNT(*) AS count FROM libraries").fetchone()
    if has_libraries and int(has_libraries["count"]) > 0:
        conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
            ("legacy_json_migrated", "1"),
        )
        conn.commit()
        return
    if TEMPLATE_LIBRARY_PATH.exists():
        try:
            raw = json.loads(TEMPLATE_LIBRARY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {"versions": []}
        payload = _normalize_template_library_payload(raw)
        for version in payload["versions"]:
            library = dict(version)
            _sqlite_insert_library_locked(conn, library)
        if payload.get("active_version_id"):
            conn.execute(
                "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
                ("active_library_id", payload["active_version_id"]),
            )
    conn.execute(
        "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
        ("legacy_json_migrated", "1"),
    )
    conn.commit()


def _sqlite_cleanup_duplicate_empty_libraries_locked(conn):
    active_id = _sqlite_active_library_id_locked(conn, allow_missing=True)
    rows = conn.execute(
        """
        SELECT
            libraries.id,
            libraries.label,
            libraries.created_at,
            libraries.updated_at,
            COUNT(templates.id) AS template_count
        FROM libraries
        LEFT JOIN templates ON templates.library_id = libraries.id
        GROUP BY libraries.id
        ORDER BY libraries.created_at ASC, libraries.id ASC
        """
    ).fetchall()
    seen = {}
    delete_ids = []
    for row in rows:
        if int(row["template_count"]) != 0:
            continue
        key = (row["label"], float(row["created_at"]), float(row["updated_at"]))
        existing_id = seen.get(key)
        if existing_id is None:
            seen[key] = row["id"]
            continue
        if row["id"] == active_id:
            delete_ids.append(existing_id)
            seen[key] = row["id"]
        else:
            delete_ids.append(row["id"])
    if not delete_ids:
        return
    conn.executemany("DELETE FROM libraries WHERE id = ?", [(id_,) for id_ in delete_ids])
    if active_id in delete_ids:
        fallback = conn.execute(
            "SELECT id FROM libraries ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if fallback is not None:
            conn.execute(
                "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
                ("active_library_id", fallback["id"]),
            )
    conn.commit()


def _sqlite_ensure_default_library_locked(conn):
    row = conn.execute("SELECT id FROM libraries LIMIT 1").fetchone()
    if row is None:
        now = time.time()
        library = {
            "id": uuid.uuid4().hex,
            "label": "Default Library",
            "created_at": now,
            "updated_at": now,
            "templates": [],
        }
        _sqlite_insert_library_locked(conn, library)
        conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
            ("active_library_id", library["id"]),
        )
        conn.commit()
        return library["id"]
    active_id = _sqlite_active_library_id_locked(conn, allow_missing=True)
    if active_id is None:
        conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
            ("active_library_id", row["id"]),
        )
        conn.commit()
        return row["id"]
    return active_id


def _sqlite_active_library_id_locked(conn, *, allow_missing=False):
    row = conn.execute(
        "SELECT value FROM app_meta WHERE key = 'active_library_id'"
    ).fetchone()
    active_id = row["value"] if row else None
    if active_id:
        exists = conn.execute(
            "SELECT id FROM libraries WHERE id = ?", (active_id,)
        ).fetchone()
        if exists is not None:
            return active_id
    if allow_missing:
        return None
    fallback = conn.execute(
        "SELECT id FROM libraries ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if fallback is None:
        return _sqlite_ensure_default_library_locked(conn)
    conn.execute(
        "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
        ("active_library_id", fallback["id"]),
    )
    conn.commit()
    return fallback["id"]


def _sqlite_set_active_library_locked(conn, library_id):
    exists = conn.execute(
        "SELECT id FROM libraries WHERE id = ?", (library_id,)
    ).fetchone()
    if exists is None:
        return False
    conn.execute(
        "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)",
        ("active_library_id", library_id),
    )
    conn.commit()
    return True


def _sqlite_template_from_row(row):
    try:
        group_center = json.loads(row["group_center_json"])
    except (TypeError, json.JSONDecodeError):
        group_center = None
    try:
        entities = json.loads(row["entities_json"])
    except (TypeError, json.JSONDecodeError):
        entities = []
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "group_center": group_center,
        "entities": entities,
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
    }


def _sqlite_templates_for_library_locked(conn, library_id):
    rows = conn.execute(
        """
        SELECT * FROM templates
        WHERE library_id = ?
        ORDER BY lower(category), lower(name), created_at
        """,
        (library_id,),
    ).fetchall()
    return [_sqlite_template_from_row(row) for row in rows]


def _sqlite_library_response():
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            active_id = _sqlite_active_library_id_locked(conn)
            active = conn.execute(
                "SELECT * FROM libraries WHERE id = ?", (active_id,)
            ).fetchone()
            library_rows = conn.execute(
                """
                SELECT libraries.*, COUNT(templates.id) AS template_count
                FROM libraries
                LEFT JOIN templates ON templates.library_id = libraries.id
                GROUP BY libraries.id
                ORDER BY libraries.created_at DESC
                """
            ).fetchall()
            templates = _sqlite_templates_for_library_locked(conn, active_id)

    libraries = [
        {
            "id": row["id"],
            "label": row["label"],
            "template_count": int(row["template_count"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "is_active": row["id"] == active_id,
        }
        for row in library_rows
    ]
    return {
        "active_library_id": active["id"],
        "active_library_label": active["label"],
        "library_count": len(libraries),
        "libraries": libraries,
        "template_count": len(templates),
        "categories": _build_template_library_categories(templates),
        "active_version_id": active["id"],
        "active_version_label": active["label"],
        "version_count": len(libraries),
        "versions": libraries,
    }


def _sqlite_create_library(label):
    now = time.time()
    library = {
        "id": uuid.uuid4().hex,
        "label": str(label or "").strip()
        or "Library " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "created_at": now,
        "updated_at": now,
        "templates": [],
    }
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            _sqlite_insert_library_locked(conn, library)
            _sqlite_set_active_library_locked(conn, library["id"])
            conn.commit()
    return library


def _sqlite_rename_library(library_id, label):
    label = str(label or "").strip()
    if not library_id or not label:
        return False
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            cur = conn.execute(
                "UPDATE libraries SET label = ?, updated_at = ? WHERE id = ?",
                (label, time.time(), library_id),
            )
            conn.commit()
            return cur.rowcount > 0


def _sqlite_select_library(library_id):
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            return _sqlite_set_active_library_locked(conn, library_id)


def _sqlite_delete_library(library_id):
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            exists = conn.execute(
                "SELECT id FROM libraries WHERE id = ?", (library_id,)
            ).fetchone()
            if exists is None:
                return False
            conn.execute("DELETE FROM libraries WHERE id = ?", (library_id,))
            fallback = conn.execute(
                "SELECT id FROM libraries ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if fallback is None:
                now = time.time()
                library = {
                    "id": uuid.uuid4().hex,
                    "label": "Default Library",
                    "created_at": now,
                    "updated_at": now,
                    "templates": [],
                }
                _sqlite_insert_library_locked(conn, library)
                fallback_id = library["id"]
            else:
                fallback_id = fallback["id"]
            _sqlite_set_active_library_locked(conn, fallback_id)
            conn.commit()
            return True


def _sqlite_save_template_library_entry(name, category, group_center, entities):
    cleaned_entities = _sanitize_template_entities(entities)
    if not cleaned_entities:
        return None
    cleaned_group_center = _sanitize_template_group_center(group_center)
    if cleaned_group_center is None:
        cleaned_group_center = _compute_group_center_from_entities(cleaned_entities)
    now = time.time()
    record = {
        "id": uuid.uuid4().hex,
        "name": str(name or "").strip() or _default_template_name(),
        "category": _normalize_template_category(category),
        "group_center": cleaned_group_center,
        "entities": cleaned_entities,
        "created_at": now,
        "updated_at": now,
    }
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            library_id = _sqlite_active_library_id_locked(conn)
            conn.execute(
                """
                INSERT INTO templates(
                    id, library_id, name, category, group_center_json, entities_json,
                    created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    library_id,
                    record["name"],
                    record["category"],
                    json.dumps(record["group_center"], ensure_ascii=False),
                    json.dumps(record["entities"], ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE libraries SET updated_at = ? WHERE id = ?",
                (now, library_id),
            )
            conn.commit()
    return record


def _sqlite_resolve_template_entries(template_ids):
    if not isinstance(template_ids, list):
        return []
    wanted = [str(tid) for tid in template_ids if str(tid).strip()]
    if not wanted:
        return []
    placeholders = ",".join("?" for _ in wanted)
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            library_id = _sqlite_active_library_id_locked(conn)
            rows = conn.execute(
                f"""
                SELECT * FROM templates
                WHERE library_id = ? AND id IN ({placeholders})
                """,
                [library_id, *wanted],
            ).fetchall()
    records = {
        _sqlite_template_from_row(row)["id"]: _sqlite_template_from_row(row)
        for row in rows
    }
    return [records[tid] for tid in wanted if tid in records]


def _sqlite_delete_template(template_id):
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            library_id = _sqlite_active_library_id_locked(conn)
            cur = conn.execute(
                "DELETE FROM templates WHERE library_id = ? AND id = ?",
                (library_id, template_id),
            )
            if cur.rowcount == 0:
                return False
            conn.execute(
                "UPDATE libraries SET updated_at = ? WHERE id = ?",
                (time.time(), library_id),
            )
            conn.commit()
            return True


def _sqlite_delete_category(category_name):
    category = _normalize_template_category(category_name)
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            library_id = _sqlite_active_library_id_locked(conn)
            cur = conn.execute(
                "DELETE FROM templates WHERE library_id = ? AND category = ?",
                (library_id, category),
            )
            if cur.rowcount == 0:
                return 0
            conn.execute(
                "UPDATE libraries SET updated_at = ? WHERE id = ?",
                (time.time(), library_id),
            )
            conn.commit()
            return cur.rowcount


def _sqlite_import_libraries(raw):
    payload = _normalize_template_library_payload(raw)
    imported = []
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            for source in payload["versions"]:
                library = dict(source)
                library["id"] = uuid.uuid4().hex
                library["templates"] = [
                    dict(template, id=uuid.uuid4().hex)
                    for template in source.get("templates", [])
                ]
                _sqlite_insert_library_locked(conn, library)
                imported.append(library)
            if imported:
                _sqlite_set_active_library_locked(conn, imported[-1]["id"])
            conn.commit()
    return imported


def _sqlite_export_active_library():
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)
            library_id = _sqlite_active_library_id_locked(conn)
            library = conn.execute(
                "SELECT * FROM libraries WHERE id = ?", (library_id,)
            ).fetchone()
            templates = _sqlite_templates_for_library_locked(conn, library_id)
    return {
        "library_id": library["id"],
        "library_label": library["label"],
        "created_at": float(library["created_at"]),
        "updated_at": float(library["updated_at"]),
        "templates": templates,
    }


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


def _clamp_pct_pair(pt):
    if not isinstance(pt, list) or len(pt) < 2:
        return None
    try:
        return [_clamp01(float(pt[0])), _clamp01(float(pt[1]))]
    except (TypeError, ValueError):
        return None


def _normalize_roi_pct(roi_pct):
    if not isinstance(roi_pct, list) or len(roi_pct) != 2:
        return None
    p1 = _clamp_pct_pair(roi_pct[0])
    p2 = _clamp_pct_pair(roi_pct[1])
    if not p1 or not p2:
        return None
    if math.isclose(p1[0], p2[0]) or math.isclose(p1[1], p2[1]):
        return None
    return [
        [min(p1[0], p2[0]), min(p1[1], p2[1])],
        [max(p1[0], p2[0]), max(p1[1], p2[1])],
    ]


# --- Region segmentation for agent proposal ---

VIEW_GRID_SIZE_DEFAULT = 32
VIEW_MIN_REGION_CELLS_DEFAULT = 4
VIEW_CLOSE_KERNEL_DEFAULT = 0
VIEW_CONNECTIVITY_DEFAULT = 4
VIEW_DENSITY_QUANTILE_DEFAULT = 0.10
# Conservative thresholds: every label other than `view` requires positive evidence.
# Chip package drawings often have no table at all; default must be `view`.
TEXT_RATIO_TABLE_THRESHOLD = 0.40
DIMENSION_RATIO_THRESHOLD = 0.30
REGION_ASPECT_TITLE_THRESHOLD = 6.0
REGION_EDGE_PROXIMITY_RATIO = 0.05
REGION_CIRCLE_RATIO_VIEW = 0.10
REGION_LABELS = {"view", "detail", "table", "title_block", "dimension", "note", "unknown"}
REGION_PIPELINE_LABELS = {"view", "detail"}
# extract_template_features folds LINE/LWPOLYLINE/ARC/POLYLINE/ELLIPSE/SPLINE into COMPOSITE_SHAPE.
REGION_STRUCTURAL_TYPES = ("COMPOSITE_SHAPE",)
REGION_TEXT_TYPES = ("TEXT", "MTEXT", "ATTRIB", "ATTDEF")
REGION_DIMENSION_TYPES = ("DIMENSION", "LEADER")
# Canonical target classes the agent pipeline can extract.
AGENT_TARGET_CLASSES = ("SMD", "Substrate", "Die area", "Alignment mark")


def _compute_density_grid(cache_payload, gx=VIEW_GRID_SIZE_DEFAULT, gy=None):
    """Bin fingerprint centers into a gx*gy density grid. Row 0 = top of render."""
    if gy is None:
        gy = gx
    bounds = cache_payload.get("bounds") or {}
    width = float(bounds.get("width") or 0.0)
    height = float(bounds.get("height") or 0.0)
    fingerprints = cache_payload.get("fingerprints") or []
    grid = np.zeros((gy, gx), dtype=np.int32)
    if width <= 0 or height <= 0 or not fingerprints:
        return {"grid": grid, "gx": gx, "gy": gy, "bounds": bounds}
    min_x = float(bounds["min_x"])
    max_y = float(bounds["max_y"])
    for f in fingerprints:
        u = (float(f.get("x", 0.0)) - min_x) / width
        v = (max_y - float(f.get("y", 0.0))) / height
        ix = min(gx - 1, max(0, int(u * gx)))
        iy = min(gy - 1, max(0, int(v * gy)))
        grid[iy, ix] += 1
    return {"grid": grid, "gx": gx, "gy": gy, "bounds": bounds}


def _morph_close_grid(mask, kernel=VIEW_CLOSE_KERNEL_DEFAULT):
    """Square-kernel morphological closing on a 2D binary grid."""
    if kernel <= 0 or mask.size == 0:
        return mask
    h, w = mask.shape
    k = int(kernel)

    def _shift_or(out, src):
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                ys = slice(max(0, dy), h + min(0, dy))
                xs = slice(max(0, dx), w + min(0, dx))
                ys_src = slice(max(0, -dy), h + min(0, -dy))
                xs_src = slice(max(0, -dx), w + min(0, -dx))
                out[ys, xs] |= src[ys_src, xs_src]

    def _shift_and(out, src):
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                ys = slice(max(0, dy), h + min(0, dy))
                xs = slice(max(0, dx), w + min(0, dx))
                ys_src = slice(max(0, -dy), h + min(0, -dy))
                xs_src = slice(max(0, -dx), w + min(0, -dx))
                shifted = np.zeros_like(src)
                shifted[ys, xs] = src[ys_src, xs_src]
                out &= shifted

    dilated = np.zeros_like(mask)
    _shift_or(dilated, mask)
    eroded = np.ones_like(dilated)
    _shift_and(eroded, dilated)
    return eroded


def _label_connected_regions(
    density,
    min_cells=VIEW_MIN_REGION_CELLS_DEFAULT,
    close_kernel=VIEW_CLOSE_KERNEL_DEFAULT,
    connectivity=VIEW_CONNECTIVITY_DEFAULT,
    density_quantile=VIEW_DENSITY_QUANTILE_DEFAULT,
):
    """Connected-component labeling on a density grid; returns list of region dicts.

    A cell is "occupied" iff its count >= the q-th quantile of all nonzero counts
    (default q=0.10 trims sparse bridge cells from dimension/border lines).
    """
    grid = density["grid"]
    gx = density["gx"]
    gy = density["gy"]
    if grid.size == 0 or grid.sum() == 0:
        return []
    nonzero = grid[grid > 0]
    if density_quantile > 0 and nonzero.size > 0:
        floor = float(np.quantile(nonzero, _clamp01(density_quantile)))
        floor = max(floor, 1.0)
    else:
        floor = 1.0
    mask = (grid >= floor).astype(np.uint8)
    if mask.sum() == 0:
        mask = (grid > 0).astype(np.uint8)
    mask = _morph_close_grid(mask, close_kernel)
    h, w = mask.shape
    if connectivity == 8:
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    labels = -np.ones((h, w), dtype=np.int32)
    next_label = 0
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] == 0 or labels[sy, sx] >= 0:
                continue
            stack = [(sy, sx)]
            labels[sy, sx] = next_label
            while stack:
                cy, cx = stack.pop()
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and labels[ny, nx] < 0
                    ):
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
            next_label += 1

    regions = []
    for label in range(next_label):
        ys, xs = np.where(labels == label)
        if ys.size < min_cells:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        entity_count = int(sum(int(grid[y, x]) for y, x in zip(ys.tolist(), xs.tolist())))
        regions.append(
            {
                "id": f"region_{label:02d}",
                "roi_pct": [[x0 / gx, y0 / gy], [x1 / gx, y1 / gy]],
                "cell_count": int(ys.size),
                "entity_count": entity_count,
                "grid_bbox": [x0, y0, x1, y1],
            }
        )
    regions.sort(key=lambda r: -r["entity_count"])
    return regions


def _region_dxf_bbox(region, bounds):
    """Convert a render-pct ROI to a DXF-space bbox."""
    (x0_pct, y0_pct), (x1_pct, y1_pct) = region["roi_pct"]
    width = float(bounds["width"])
    height = float(bounds["height"])
    dxf_min_x = bounds["min_x"] + x0_pct * width
    dxf_max_x = bounds["min_x"] + x1_pct * width
    dxf_max_y = bounds["max_y"] - y0_pct * height
    dxf_min_y = bounds["max_y"] - y1_pct * height
    return dxf_min_x, dxf_min_y, dxf_max_x, dxf_max_y


def _region_features(cache_payload, region):
    """Compute geometry-only features for a region (no LLM)."""
    bounds = cache_payload["bounds"]
    fingerprints = cache_payload.get("fingerprints") or []
    dxf_min_x, dxf_min_y, dxf_max_x, dxf_max_y = _region_dxf_bbox(region, bounds)
    type_counts = {}
    sizes = []
    members = 0
    for f in fingerprints:
        x = float(f.get("x", 0.0))
        y = float(f.get("y", 0.0))
        if not (dxf_min_x <= x <= dxf_max_x and dxf_min_y <= y <= dxf_max_y):
            continue
        members += 1
        ftype = f.get("type") or "UNKNOWN"
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
        sz = float(f.get("size") or 0.0)
        if sz > 0:
            sizes.append(sz)

    rwidth = max(dxf_max_x - dxf_min_x, BOUNDS_EPS)
    rheight = max(dxf_max_y - dxf_min_y, BOUNDS_EPS)
    aspect = max(rwidth, rheight) / min(rwidth, rheight)
    bw = max(float(bounds["width"]), BOUNDS_EPS)
    bh = max(float(bounds["height"]), BOUNDS_EPS)
    area_ratio = (rwidth * rheight) / (bw * bh)
    edge_proximity = min(
        (dxf_min_x - bounds["min_x"]) / bw,
        (bounds["max_x"] - dxf_max_x) / bw,
        (dxf_min_y - bounds["min_y"]) / bh,
        (bounds["max_y"] - dxf_max_y) / bh,
    )
    if members > 0:
        type_dist = {t: c / members for t, c in type_counts.items()}
    else:
        type_dist = {}
    structural_share = sum(type_dist.get(t, 0.0) for t in REGION_STRUCTURAL_TYPES)
    text_share = sum(type_dist.get(t, 0.0) for t in REGION_TEXT_TYPES)
    dimension_share = sum(type_dist.get(t, 0.0) for t in REGION_DIMENSION_TYPES)
    circle_share = type_dist.get("CIRCLE", 0.0)
    composite_share = type_dist.get("COMPOSITE_SHAPE", 0.0)

    repetition_entropy = 1.0
    if sizes:
        arr = np.asarray(sizes, dtype=np.float64)
        if arr.size > 1 and arr.max() > 0:
            buckets = np.histogram(np.log10(arr.clip(min=1e-9)), bins=20)[0]
            total = buckets.sum()
            if total > 0:
                probs = buckets / total
                nz = probs[probs > 0]
                if nz.size > 0:
                    ent = float(-(nz * np.log2(nz)).sum())
                    repetition_entropy = ent / math.log2(max(2, nz.size))

    return {
        "entity_count": members,
        "type_dist": {t: round(v, 4) for t, v in type_dist.items()},
        "structural_share": round(structural_share, 4),
        "circle_share": round(circle_share, 4),
        "text_share": round(text_share, 4),
        "dimension_share": round(dimension_share, 4),
        "composite_share": round(composite_share, 4),
        "size_mean": round(float(np.mean(sizes)) if sizes else 0.0, 4),
        "size_std": round(float(np.std(sizes)) if sizes else 0.0, 4),
        "aspect_ratio": round(float(aspect), 3),
        "area_ratio": round(float(area_ratio), 4),
        "edge_proximity": round(float(max(edge_proximity, 0.0)), 4),
        "repetition_entropy": round(float(repetition_entropy), 4),
        "dxf_bbox": [
            round(dxf_min_x, 3),
            round(dxf_min_y, 3),
            round(dxf_max_x, 3),
            round(dxf_max_y, 3),
        ],
    }


def _classify_region_heuristic(features):
    """Assign a label using only geometric features.

    Default is `view`. Other labels only fire on positive evidence — chip
    package drawings often have no table or title block at all, so we never
    flip away from `view` on the absence of pad density alone.

    Fingerprint type taxonomy in this codebase:
      - CIRCLE: SMD pads, alignment marks
      - COMPOSITE_SHAPE: every line/polyline/arc/ellipse/spline (package outline,
        substrate traces, table grid lines, dimension shafts all collapse here)
      - TEXT/MTEXT/ATTRIB: notes, table contents, title block labels
      - DIMENSION/LEADER: dimension annotations
    """
    if features["entity_count"] <= 0:
        return {"label": "unknown", "confidence": 0.0, "reasons": ["empty_region"]}
    aspect = features["aspect_ratio"]
    edge = features["edge_proximity"]
    circle_share = features["circle_share"]
    text_share = features["text_share"]
    dimension_share = features["dimension_share"]
    composite_share = features["composite_share"]
    repetition = features["repetition_entropy"]
    area = features["area_ratio"]

    # Dimension strip: leader/dimension entities dominate, OR the region is a
    # very thin strip near an edge with no pads or text.
    if dimension_share >= DIMENSION_RATIO_THRESHOLD:
        return {
            "label": "dimension",
            "confidence": 0.7,
            "reasons": ["dimension_or_leader_dominated"],
        }
    if (
        aspect >= REGION_ASPECT_TITLE_THRESHOLD * 1.5
        and area < 0.05
        and edge <= REGION_EDGE_PROXIMITY_RATIO
        and circle_share < 0.02
    ):
        return {
            "label": "dimension",
            "confidence": 0.55,
            "reasons": ["very_thin_edge_strip"],
        }
    # Note / table: text dominance only. We do NOT flip line-dominated regions
    # to table — many real package views are mostly polylines (substrate
    # traces, alignment marks), and false-positive tables hide them.
    if text_share >= TEXT_RATIO_TABLE_THRESHOLD:
        return {
            "label": "note" if area < 0.04 else "table",
            "confidence": 0.7,
            "reasons": ["high_text_share"],
        }
    # Title block: thin region, hugs an edge, mostly lines, no pads.
    if (
        aspect >= REGION_ASPECT_TITLE_THRESHOLD
        and edge <= REGION_EDGE_PROXIMITY_RATIO
        and area < 0.10
        and circle_share < 0.02
        and composite_share >= 0.5
    ):
        return {
            "label": "title_block",
            "confidence": 0.65,
            "reasons": ["thin_edge_hugging_lines"],
        }
    # Default: view. Score reflects how confident we are this is a real
    # candidate region for one of the 4 target classes (SMD, Substrate,
    # Die area, Alignment mark) — circle/pad density and repetition help.
    reasons = []
    confidence = 0.55
    if circle_share >= REGION_CIRCLE_RATIO_VIEW:
        reasons.append("circle_pad_density")
        confidence = max(confidence, 0.8)
    if repetition < 0.6:
        reasons.append("low_size_entropy_repetition")
        confidence = max(confidence, 0.72)
    if composite_share >= 0.3 and circle_share < 0.02:
        reasons.append("substrate_or_outline_lines")
        confidence = max(confidence, 0.6)
    if not reasons:
        reasons.append("default_view")
    return {"label": "view", "confidence": round(confidence, 3), "reasons": reasons}


def _classify_regions_heuristic(regions_with_features):
    classified = []
    for region in regions_with_features:
        verdict = _classify_region_heuristic(region["features"])
        merged = dict(region)
        merged["label"] = verdict["label"]
        merged["label_confidence"] = verdict["confidence"]
        merged["label_reasons"] = verdict["reasons"]
        merged["label_source"] = "heuristic"
        merged["included_in_pipeline"] = verdict["label"] in REGION_PIPELINE_LABELS
        classified.append(merged)
    return classified


def _format_target_class_block(target_classes, target_descriptions):
    """Render the user's target classes + descriptions for prompt injection."""
    target_classes = list(target_classes or [])
    desc_map = target_descriptions if isinstance(target_descriptions, dict) else {}
    if not target_classes and not desc_map:
        return "(user did not specify target classes; treat any candidate region as a view)"
    lines = []
    seen = set()
    for cls in target_classes:
        seen.add(cls)
        desc = (desc_map.get(cls) or "").strip()
        lines.append(f"  - {cls}" + (f": {desc}" if desc else ""))
    for cls, desc in desc_map.items():
        if cls in seen:
            continue
        desc = (desc or "").strip()
        lines.append(f"  - {cls}" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


def _agent_region_classifier_prompt(
    regions, instruction, target_classes, target_descriptions=None
):
    summary = [
        {
            "id": r["id"],
            "roi_pct": r["roi_pct"],
            "heuristic_label": r["label"],
            "heuristic_confidence": r["label_confidence"],
            "features": r["features"],
        }
        for r in regions
    ]
    target_block = _format_target_class_block(target_classes, target_descriptions)
    return (
        "You refine region labels for a CAD drawing region map.\n"
        "The user wants to extract these target classes (descriptions provided):\n"
        f"{target_block}\n"
        "A 'view' is any region likely to contain instances of those targets.\n"
        "Each region has a render-pct bbox in [0,1] and numeric features.\n"
        "Allowed labels: view, detail, table, title_block, dimension, note, unknown.\n"
        "DEFAULT to 'view' unless the region clearly is a non-view artifact:\n"
        "  - 'table': only if text_share is high AND structural lines form a grid;\n"
        "             not every drawing has a table — do not invent one\n"
        "  - 'title_block': thin strip hugging a drawing edge, mostly lines\n"
        "  - 'dimension': dimension/leader dominated OR a thin strip with no targets\n"
        "  - 'note': small text-heavy region NOT in a grid\n"
        "  - 'detail': a zoomed callout, usually labelled\n"
        "When in doubt label 'view' — false negatives lose target candidates.\n"
        "Return strict JSON without markdown.\n"
        "Schema:\n"
        '{"regions":[{"id":"region_00","label":"view","name":"top_view",'
        '"confidence":0.0,"reasons":["..."]}]}\n'
        f"User instruction: {instruction or ''}\n"
        f"Regions: {json.dumps(summary)}\n"
    )


def _classify_regions_llm(
    regions_with_features,
    *,
    instruction,
    target_classes,
    target_descriptions=None,
    provider,
    raise_on_error=False,
):
    """Optionally refine heuristic labels with an LLM. Defaults to silent fallback;
    pass raise_on_error=True to propagate provider errors (used by /agent/propose).

    Region classification is text-only — uses llm_model only, never vlm_model
    (vision waste for numeric features)."""
    if not regions_with_features:
        return regions_with_features
    config = _vllm_config(provider)
    model = config["llm_model"]
    if not config["enabled"] or not model:
        return regions_with_features
    prompt = _agent_region_classifier_prompt(
        regions_with_features,
        instruction,
        target_classes,
        target_descriptions=target_descriptions,
    )
    messages = [
        {
            "role": "system",
            "content": "You return strict JSON for CAD region classification.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        raw = _vllm_chat_completion_with_retry(model, messages, provider=provider)
    except RuntimeError:
        if raise_on_error:
            raise
        return regions_with_features
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        return regions_with_features
    overrides = {}
    for item in parsed.get("regions") or []:
        if isinstance(item, dict) and item.get("id"):
            overrides[item["id"]] = item

    refined = []
    for region in regions_with_features:
        merged = dict(region)
        ov = overrides.get(region["id"])
        if isinstance(ov, dict):
            label = str(ov.get("label") or merged["label"]).strip().lower()
            if label in REGION_LABELS:
                merged["label"] = label
            try:
                merged["label_confidence"] = round(
                    _clamp01(float(ov.get("confidence"))), 3
                )
            except (TypeError, ValueError):
                pass
            reasons = ov.get("reasons")
            if isinstance(reasons, list):
                merged["label_reasons"] = [str(x) for x in reasons[:5]]
            name = ov.get("name")
            if isinstance(name, str) and name.strip():
                merged["view_name"] = name.strip()
            merged["label_source"] = "llm"
        merged["included_in_pipeline"] = merged["label"] in REGION_PIPELINE_LABELS
        refined.append(merged)
    return refined


def _detect_regions(
    cache_payload,
    *,
    instruction=None,
    target_classes=None,
    target_descriptions=None,
    provider=None,
    settings=None,
    raise_on_llm_error=False,
):
    """Run density grid -> components -> features -> heuristic -> optional LLM refine."""
    settings = settings or {}
    grid_size = int(settings.get("view_grid_size") or VIEW_GRID_SIZE_DEFAULT)
    min_cells = int(
        settings.get("view_min_region_cells") or VIEW_MIN_REGION_CELLS_DEFAULT
    )
    close_kernel = int(settings.get("view_close_kernel", VIEW_CLOSE_KERNEL_DEFAULT))
    connectivity = int(
        settings.get("view_connectivity") or VIEW_CONNECTIVITY_DEFAULT
    )
    density_quantile = float(
        settings.get("view_density_quantile", VIEW_DENSITY_QUANTILE_DEFAULT)
    )
    density = _compute_density_grid(cache_payload, gx=grid_size, gy=grid_size)
    raw_regions = _label_connected_regions(
        density,
        min_cells=min_cells,
        close_kernel=close_kernel,
        connectivity=connectivity,
        density_quantile=density_quantile,
    )
    enriched = []
    for region in raw_regions:
        features = _region_features(cache_payload, region)
        enriched.append(
            {
                "id": region["id"],
                "roi_pct": region["roi_pct"],
                "grid_bbox": region["grid_bbox"],
                "cell_count": region["cell_count"],
                "entity_count": region["entity_count"],
                "features": features,
            }
        )
    classified = _classify_regions_heuristic(enriched)
    refined = _classify_regions_llm(
        classified,
        instruction=instruction or "",
        target_classes=target_classes or [],
        target_descriptions=target_descriptions or {},
        provider=provider,
        raise_on_error=raise_on_llm_error,
    )
    view_idx = 0
    for region in refined:
        if region["label"] in REGION_PIPELINE_LABELS and not region.get("view_name"):
            view_idx += 1
            region["view_name"] = f"view_{view_idx:02d}"
    return refined, density


# --- Tile rendering for VLM seed proposal ---

TILE_PX_LONG_SIDE_DEFAULT = 768
TILE_GRID_PER_VIEW_DEFAULT = 3
SEED_DEDUP_DIST_PCT_DEFAULT = 0.01


def _render_tile_png(
    cache_payload,
    roi_pct,
    *,
    px_long_side=TILE_PX_LONG_SIDE_DEFAULT,
    line_width=0.6,
    dpi=100,
):
    """Render fingerprint geometry inside roi_pct to PNG bytes (lazy matplotlib)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import io as _io

    bounds = cache_payload["bounds"]
    fingerprints = cache_payload.get("fingerprints") or []
    (x0p, y0p), (x1p, y1p) = roi_pct
    width = float(bounds["width"])
    height = float(bounds["height"])
    dxf_min_x = bounds["min_x"] + x0p * width
    dxf_max_x = bounds["min_x"] + x1p * width
    dxf_max_y = bounds["max_y"] - y0p * height
    dxf_min_y = bounds["max_y"] - y1p * height
    tile_w = max(dxf_max_x - dxf_min_x, BOUNDS_EPS)
    tile_h = max(dxf_max_y - dxf_min_y, BOUNDS_EPS)
    aspect = tile_w / tile_h
    if aspect >= 1:
        fig_w_px = int(px_long_side)
        fig_h_px = max(64, int(round(px_long_side / aspect)))
    else:
        fig_h_px = int(px_long_side)
        fig_w_px = max(64, int(round(px_long_side * aspect)))

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(dxf_min_x, dxf_max_x)
    ax.set_ylim(dxf_min_y, dxf_max_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    margin_x = tile_w * 1e-3
    margin_y = tile_h * 1e-3
    for f in fingerprints:
        cx = float(f.get("x", 0.0))
        cy = float(f.get("y", 0.0))
        if not (
            dxf_min_x - margin_x <= cx <= dxf_max_x + margin_x
            and dxf_min_y - margin_y <= cy <= dxf_max_y + margin_y
        ):
            continue
        geom = f.get("geometry") or {}
        kind = geom.get("kind")
        if kind == "circle":
            ax.add_patch(
                plt.Circle(
                    (geom.get("cx", cx), geom.get("cy", cy)),
                    float(geom.get("r", f.get("size", 0.0))),
                    fill=False,
                    linewidth=line_width,
                    edgecolor="black",
                )
            )
        elif kind == "polyline":
            pts = geom.get("points") or []
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, linewidth=line_width, color="black")
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _iter_tiles_for_view(roi_pct, *, grid_x=TILE_GRID_PER_VIEW_DEFAULT, grid_y=None):
    """Yield (tile_index, tile_roi_pct, local_origin_pct, local_size_pct).

    local_origin_pct/local_size_pct describe the tile bbox within the parent ROI,
    so VLM-returned tile-local coords can be mapped back to full-render pct.
    """
    if grid_y is None:
        grid_y = grid_x
    grid_x = max(1, int(grid_x))
    grid_y = max(1, int(grid_y))
    (x0, y0), (x1, y1) = roi_pct
    rw = x1 - x0
    rh = y1 - y0
    tw = rw / grid_x
    th = rh / grid_y
    idx = 0
    for iy in range(grid_y):
        for ix in range(grid_x):
            tx0 = x0 + ix * tw
            ty0 = y0 + iy * th
            tx1 = tx0 + tw
            ty1 = ty0 + th
            yield idx, [[tx0, ty0], [tx1, ty1]], (tx0, ty0), (tw, th)
            idx += 1


def _tile_local_to_full_pct(local_xy, tile_origin, tile_size):
    """Map [u, v] in tile-local pct ([0,1]) back to full-render pct."""
    return [
        _clamp01(tile_origin[0] + local_xy[0] * tile_size[0]),
        _clamp01(tile_origin[1] + local_xy[1] * tile_size[1]),
    ]


def _agent_seed_tile_prompt(
    *,
    instruction,
    target_classes,
    target_descriptions=None,
    view_name,
    tile_index,
):
    target_block = _format_target_class_block(target_classes, target_descriptions)
    return (
        "You are looking at a tile cropped from a CAD drawing.\n"
        "The tile is rendered in greyscale; black strokes are CAD geometry.\n"
        "The user wants to find these target classes (descriptions provided):\n"
        f"{target_block}\n"
        "For each target you SEE in this tile, propose ONE seed candidate.\n"
        "Coordinates you return MUST be tile-local render percentages in [0,1].\n"
        "(0,0) is the top-left of THIS tile. Do not use full-drawing coords.\n"
        "Use polygon_pct for repeated multi-entity footprints (3+ points).\n"
        "Use click_pct for a single small element such as one pad.\n"
        "If the tile contains no clear instance of any target class, return seeds: [].\n"
        "Return only JSON, no markdown.\n"
        "Schema:\n"
        '{"seeds":[{"target_class":"<one of the user target classes>",'
        '"label":"...","polygon_pct":[[x,y],[x,y],[x,y]],'
        '"click_pct":[x,y],"confidence":0.0}],"notes":["..."]}\n'
        f"Tile view: {view_name} (tile #{tile_index})\n"
        f"User instruction: {instruction or ''}\n"
    )


def _normalize_tile_seeds(parsed, *, tile_origin, tile_size, view_name, tile_index):
    if not isinstance(parsed, dict):
        return []
    seeds = []
    for item in parsed.get("seeds") or []:
        if not isinstance(item, dict):
            continue
        target_class = str(item.get("target_class") or "SMD").strip() or "SMD"
        confidence = item.get("confidence")
        try:
            confidence = round(_clamp01(float(confidence)), 3) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        seed = {
            "target_class": target_class,
            "view_name": view_name,
            "label": str(item.get("label") or "").strip() or None,
            "confidence": confidence,
            "source": "vlm_tile",
            "tile_index": tile_index,
        }
        click_local = _clamp_pct_pair(item.get("click_pct"))
        if click_local:
            seed["click_pct"] = _tile_local_to_full_pct(
                click_local, tile_origin, tile_size
            )
        polygon_local = item.get("polygon_pct")
        if isinstance(polygon_local, list) and len(polygon_local) >= 3:
            pts = [_clamp_pct_pair(p) for p in polygon_local]
            pts = [p for p in pts if p]
            if len(pts) >= 3:
                seed["polygon_pct"] = [
                    _tile_local_to_full_pct(p, tile_origin, tile_size) for p in pts
                ]
        if seed.get("click_pct") or seed.get("polygon_pct"):
            seeds.append(seed)
    return seeds


def _seed_centroid_pct(seed):
    if seed.get("click_pct"):
        return tuple(seed["click_pct"])
    poly = seed.get("polygon_pct") or []
    if poly:
        return (
            sum(p[0] for p in poly) / len(poly),
            sum(p[1] for p in poly) / len(poly),
        )
    return None


def _dedup_seeds_by_geometry(seeds, *, dist_threshold=SEED_DEDUP_DIST_PCT_DEFAULT):
    """Greedy spatial dedup: drop seeds whose centroid is within dist_threshold
    pct of an already-kept seed of the same target_class."""
    kept = []
    for seed in sorted(
        seeds,
        key=lambda s: -(s.get("confidence") if s.get("confidence") is not None else 0.0),
    ):
        c = _seed_centroid_pct(seed)
        if c is None:
            continue
        duplicate = False
        for prior in kept:
            if prior.get("target_class") != seed.get("target_class"):
                continue
            pc = _seed_centroid_pct(prior)
            if pc is None:
                continue
            if math.hypot(c[0] - pc[0], c[1] - pc[1]) <= dist_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(seed)
    return kept


def _propose_seeds_for_region(
    cache_payload,
    region,
    *,
    instruction,
    target_classes,
    target_descriptions=None,
    provider,
    settings,
):
    """Tile a view region, render each tile, ask VLM for seeds, merge and dedup.

    Returns (seeds, tile_debug, errors). Empty seeds is fine; errors lists
    per-tile failures so the caller can decide whether to surface 502.
    """
    config = _vllm_config(provider)
    vlm_model = config["vlm_model"]
    if not config["enabled"] or not vlm_model:
        return [], [], []
    grid = int(
        settings.get("tile_grid_per_view") or TILE_GRID_PER_VIEW_DEFAULT
    )
    px_long = int(
        settings.get("tile_px_long_side") or TILE_PX_LONG_SIDE_DEFAULT
    )
    dedup_dist = float(
        settings.get("seed_dedup_dist_pct") or SEED_DEDUP_DIST_PCT_DEFAULT
    )
    view_name = region.get("view_name") or region.get("id")
    seeds = []
    tile_debug = []
    errors = []
    for tile_idx, tile_roi, tile_origin, tile_size in _iter_tiles_for_view(
        region["roi_pct"], grid_x=grid
    ):
        png_bytes = _render_tile_png(
            cache_payload, tile_roi, px_long_side=px_long
        )
        import base64 as _base64

        data_url = "data:image/png;base64," + _base64.b64encode(png_bytes).decode(
            "ascii"
        )
        prompt = _agent_seed_tile_prompt(
            instruction=instruction,
            target_classes=target_classes,
            target_descriptions=target_descriptions,
            view_name=view_name,
            tile_index=tile_idx,
        )
        messages = [
            {
                "role": "system",
                "content": "You return strict JSON for CAD tile seed proposals.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        try:
            raw, used_text_only = _vllm_chat_completion_with_image_fallback(
                vlm_model, messages, provider=provider
            )
        except RuntimeError as exc:
            errors.append(
                {
                    "view_name": view_name,
                    "tile_index": tile_idx,
                    "error": str(exc),
                }
            )
            tile_debug.append(
                {
                    "view_name": view_name,
                    "tile_index": tile_idx,
                    "tile_roi_pct": tile_roi,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        parsed = _extract_json_object(raw)
        tile_seeds = _normalize_tile_seeds(
            parsed,
            tile_origin=tile_origin,
            tile_size=tile_size,
            view_name=view_name,
            tile_index=tile_idx,
        )
        if used_text_only:
            for seed in tile_seeds:
                seed["source"] = "llm_tile_text_only"
                if seed.get("confidence") is not None:
                    seed["confidence"] = round(min(seed["confidence"], 0.3), 3)
                else:
                    seed["confidence"] = 0.2
        seeds.extend(tile_seeds)
        tile_debug.append(
            {
                "view_name": view_name,
                "tile_index": tile_idx,
                "tile_roi_pct": tile_roi,
                "status": "ok_text_only" if used_text_only else "ok",
                "seed_count": len(tile_seeds),
            }
        )
    return _dedup_seeds_by_geometry(seeds, dist_threshold=dedup_dist), tile_debug, errors


def _default_agent_views():
    return [
        {
            "name": "full_drawing",
            "roi_pct": [[0.0, 0.0], [1.0, 1.0]],
            "confidence": 0.0,
            "source": "fallback",
            "needs_review": True,
        }
    ]


def _normalize_agent_views(view_models):
    views = []
    for view in view_models or []:
        roi_pct = _normalize_roi_pct(view.roi_pct)
        views.append(
            {
                "name": view.name,
                "roi_pct": roi_pct,
                "confidence": view.confidence,
                "source": view.source or "request",
                "needs_review": roi_pct is None,
            }
        )
    return views or _default_agent_views()


def _agent_view_by_name(views):
    return {view["name"]: view for view in views if view.get("name")}


def _agent_seed_payload(seed):
    return {
        "target_class": seed.target_class,
        "view_name": seed.view_name,
        "label": seed.label,
        "template_id": seed.template_id,
        "entity_count": len(seed.entities or []),
        "group_center": seed.group_center,
        "click_pct": seed.click_pct,
        "polygon_pct": seed.polygon_pct,
        "confidence": seed.confidence,
        "source": seed.source or "request",
    }


def _agent_validation_summary(extract_result, scan_result, seed_confidence=None):
    entity_count = int((extract_result or {}).get("entity_count") or 0)
    match_count = int((scan_result or {}).get("match_count") or 0)
    scan_stats = (scan_result or {}).get("scan_stats") or {}
    score_max = scan_stats.get("score_max")
    scores = [
        m.get("match_score")
        for m in (scan_result or {}).get("matches", [])
        if m.get("match_score") is not None
    ]
    mean_score = round(float(sum(scores) / len(scores)), 6) if scores else None
    warnings = []
    if entity_count <= 0:
        warnings.append("seed_extracted_no_entities")
    if match_count <= 0:
        warnings.append("scan_found_no_matches")
    if match_count > 1000:
        warnings.append("scan_found_many_matches")
    if seed_confidence is not None and seed_confidence < 0.5:
        warnings.append("low_seed_confidence")
    confidence_parts = []
    if seed_confidence is not None:
        confidence_parts.append(max(0.0, min(float(seed_confidence), 1.0)))
    if entity_count > 0:
        confidence_parts.append(0.75)
    if match_count > 0:
        confidence_parts.append(0.8)
    if warnings:
        confidence_parts.append(0.45)
    confidence = (
        round(sum(confidence_parts) / len(confidence_parts), 3)
        if confidence_parts
        else 0.0
    )
    return {
        "confidence": confidence,
        "warnings": warnings,
        "entity_count": entity_count,
        "match_count": match_count,
        "mean_match_score": mean_score,
        "score_max": score_max,
    }


def _group_agent_results(class_results):
    grouped = {}
    for result in class_results:
        target_class = result.get("target_class") or "Uncategorized"
        view_name = result.get("view_name") or "unassigned_view"
        grouped.setdefault(target_class, {}).setdefault(view_name, []).append(result)
    return grouped


def _json_response_payload(response):
    if not isinstance(response, JSONResponse):
        return response
    try:
        return json.loads(response.body.decode("utf-8"))
    except Exception:
        return {"error": "Unable to decode JSON response."}


def _provider_value(provider, attr, env_name, default=""):
    if provider is not None:
        value = getattr(provider, attr, None)
        if value is not None:
            return value
    return os.environ.get(env_name) or default


def _vllm_config(provider=None):
    base_url = str(
        _provider_value(provider, "base_url", "VLLM_BASE_URL", "")
    ).strip().rstrip("/")
    llm_model = str(
        _provider_value(provider, "llm_model", "VLLM_LLM_MODEL", "")
    ).strip()
    vlm_model = str(
        _provider_value(provider, "vlm_model", "VLLM_VLM_MODEL", "")
    ).strip()
    api_key = str(_provider_value(provider, "api_key", "VLLM_API_KEY", "")).strip()
    timeout_default = 30 if provider is not None else 60
    timeout_raw = _provider_value(
        provider, "timeout_seconds", "VLLM_TIMEOUT_SECONDS", timeout_default
    )
    try:
        timeout_seconds = float(timeout_raw)
    except (TypeError, ValueError):
        timeout_seconds = 60.0
    temperature_raw = _provider_value(provider, "temperature", "VLLM_TEMPERATURE", 0)
    try:
        temperature = float(temperature_raw)
    except (TypeError, ValueError):
        temperature = 0.0
    max_tokens_raw = _provider_value(provider, "max_tokens", "VLLM_MAX_TOKENS", 1200)
    try:
        max_tokens = int(max_tokens_raw)
    except (TypeError, ValueError):
        max_tokens = 1200
    max_tokens = max(128, min(max_tokens, 8192))
    return {
        "base_url": base_url,
        "api_key": api_key,
        "api_key_configured": bool(api_key),
        "llm_model": llm_model,
        "vlm_model": vlm_model,
        "timeout_seconds": timeout_seconds,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "send_image": bool(provider and provider.send_image),
        "enabled": bool(base_url and (vlm_model or llm_model)),
        "source": "request" if provider and provider.base_url else "environment",
    }


def _vllm_chat_url(base_url):
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def _extract_json_object(text):
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


def _vllm_chat_completion(model, messages, *, temperature=0.0, provider=None):
    config = _vllm_config(provider)
    if not config["base_url"] or not model:
        raise RuntimeError("vLLM is not configured.")
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": config["max_tokens"],
    }
    headers = {"Content-Type": "application/json"}
    api_key = config["api_key"]
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        _vllm_chat_url(config["base_url"]),
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config["timeout_seconds"]) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM connection failed: {exc}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Provider request timed out after {config['timeout_seconds']} seconds."
        ) from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("vLLM response did not include choices.")
    return ((choices[0].get("message") or {}).get("content") or "").strip()


def _vllm_chat_completion_with_retry(model, messages, *, provider=None):
    config = _vllm_config(provider)
    try:
        return _vllm_chat_completion(
            model, messages, temperature=config["temperature"], provider=provider
        )
    except RuntimeError as exc:
        message = str(exc)
        if (
            "invalid temperature" in message.lower()
            and "only 1 is allowed" in message.lower()
            and not math.isclose(config["temperature"], 1.0)
        ):
            return _vllm_chat_completion(
                model, messages, temperature=1.0, provider=provider
            )
        raise


def _messages_without_image_inputs(messages):
    stripped = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            new_message = dict(message)
            new_message["content"] = "\n".join(text_parts).strip()
            stripped.append(new_message)
        else:
            stripped.append(message)
    return stripped


_IMAGE_REJECTION_HINTS = (
    "unsupported image format",
    "invalid image",
    "image format",
    "image_url",
    "image url",
    "vision",
    "multimodal",
    "non-text",
    "content type",
    "model does not support",
    "does not support image",
    "no support for image",
    "cannot process image",
    "image input",
)


def _looks_like_image_rejection(message):
    return any(hint in message for hint in _IMAGE_REJECTION_HINTS)


def _vllm_chat_completion_with_image_fallback(model, messages, *, provider=None):
    """Call provider; if it rejects image input, transparently retry text-only.

    Returns (content, used_text_only_fallback) so callers can flag the result."""
    try:
        return _vllm_chat_completion_with_retry(model, messages, provider=provider), False
    except RuntimeError as exc:
        message = str(exc).lower()
        has_image = any(
            isinstance(msg.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in msg.get("content")
            )
            for msg in messages
        )
        if has_image and _looks_like_image_rejection(message):
            content = _vllm_chat_completion_with_retry(
                model, _messages_without_image_inputs(messages), provider=provider
            )
            return content, True
        raise


def _provider_public_debug(config):
    return {
        "base_url": config["base_url"],
        "source": config["source"],
        "api_key_configured": config["api_key_configured"],
        "llm_model": config["llm_model"] or None,
        "vlm_model": config["vlm_model"] or None,
        "timeout_seconds": config["timeout_seconds"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "send_image": config["send_image"],
    }


def _is_supported_agent_image_data_url(value):
    if not isinstance(value, str):
        return False
    prefix = value.split(",", 1)[0].lower()
    return prefix.startswith("data:image/jpeg;base64") or prefix.startswith(
        "data:image/png;base64"
    )


def _agent_proposal_prompt(request, cache_payload):
    target_classes = request.target_classes or ["SMD"]
    existing_views = [
        {
            "name": view.name,
            "roi_pct": view.roi_pct,
            "confidence": view.confidence,
            "source": view.source,
        }
        for view in request.views
    ]
    return (
        "You are proposing CAD matching review inputs for a semiconductor package drawing.\n"
        "Return only a JSON object. Do not include markdown.\n"
        "Coordinates must be render percentage coordinates in [0, 1], not pixels.\n"
        "Detect the two main package views when visible: top_view and bottom_view.\n"
        "For each requested target class, propose one or more seed candidates inside a view.\n"
        "Prefer polygon_pct for multi-entity footprints and click_pct for a single obvious entity.\n"
        "If uncertain, still return the best candidates with lower confidence and notes.\n"
        "Schema:\n"
        "{"
        "\"views\":[{\"name\":\"top_view\",\"roi_pct\":[[x1,y1],[x2,y2]],\"confidence\":0.0,\"source\":\"vlm\"}],"
        "\"seed_candidates\":[{\"target_class\":\"SMD\",\"view_name\":\"top_view\",\"label\":\"...\","
        "\"polygon_pct\":[[x,y],[x,y],[x,y]],\"click_pct\":[x,y],\"confidence\":0.0,\"source\":\"vlm\"}],"
        "\"notes\":[\"...\"]"
        "}\n"
        f"User instruction: {request.instruction}\n"
        f"Target classes: {target_classes}\n"
        f"Existing human/model views: {existing_views}\n"
        f"Drawing bounds: {cache_payload.get('bounds')}\n"
    )


def _normalize_agent_proposal(raw):
    data = raw if isinstance(raw, dict) else {}
    views = []
    for item in data.get("views") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        roi_pct = _normalize_roi_pct(item.get("roi_pct"))
        if not name or not roi_pct:
            continue
        views.append(
            {
                "name": name,
                "roi_pct": roi_pct,
                "confidence": item.get("confidence"),
                "source": item.get("source") or "vlm",
                "needs_review": True,
            }
        )
    seeds = []
    for item in data.get("seed_candidates") or []:
        if not isinstance(item, dict):
            continue
        seed = {
            "target_class": str(item.get("target_class") or "SMD"),
            "view_name": item.get("view_name"),
            "label": item.get("label"),
            "confidence": item.get("confidence"),
            "source": item.get("source") or "vlm",
        }
        click_pct = _clamp_pct_pair(item.get("click_pct"))
        polygon_pct = item.get("polygon_pct")
        if click_pct:
            seed["click_pct"] = click_pct
        if isinstance(polygon_pct, list) and len(polygon_pct) >= 3:
            pts = [_clamp_pct_pair(pt) for pt in polygon_pct]
            pts = [pt for pt in pts if pt]
            if len(pts) >= 3:
                seed["polygon_pct"] = pts
        if seed.get("click_pct") or seed.get("polygon_pct"):
            seeds.append(seed)
    return {
        "views": views,
        "seed_candidates": seeds,
        "notes": data.get("notes") if isinstance(data.get("notes"), list) else [],
    }


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


@app.on_event("startup")
async def startup_template_library_db():
    with _template_library_lock:
        with _sqlite_library_connect() as conn:
            _sqlite_library_init_locked(conn)


# --- FastAPI routes ---


@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse(
        INDEX_HTML_PATH,
        media_type="text/html",
        headers=NO_STORE_HEADERS,
    )


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


@app.get("/agent/schema")
async def get_agent_schema():
    return {
        "status": "experimental",
        "vllm": {
            "env": [
                "VLLM_BASE_URL",
                "VLLM_VLM_MODEL",
                "VLLM_LLM_MODEL",
                "VLLM_API_KEY",
                "VLLM_TIMEOUT_SECONDS",
                "VLLM_TEMPERATURE",
            ],
            "request_provider": {
                "base_url": "OpenAI-compatible base URL, e.g. https://api.moonshot.ai",
                "api_key": "Optional bearer token.",
                "llm_model": "Text model for JSON proposals.",
                "vlm_model": "Vision model for SVG/image proposals.",
                "timeout_seconds": 60,
                "temperature": "Defaults to 0. Some providers/models require 1.",
                "max_tokens": 1200,
                "send_image": "Defaults to false. When true, the frontend sends a viewer-sized JPG/PNG screenshot.",
            },
            "openai_compatible_endpoint": "/v1/chat/completions",
        },
        "target_classes": "Free-form list chosen by the user; common chip examples: SMD, Substrate, Die area, Alignment mark.",
        "target_descriptions": "Optional dict mapping each target class name to a free-form description fed into LLM/VLM prompts.",
        "request": {
            "cache_id": "Upload cache id returned by /upload.",
            "instruction": "Natural-language extraction goal.",
            "target_classes": ["SMD"],
            "target_descriptions": {
                "SMD": "small repeated circular pads arranged in arrays",
            },
            "views": [
                {
                    "name": "top_view",
                    "roi_pct": [[0.08, 0.12], [0.48, 0.82]],
                    "confidence": 0.8,
                    "source": "vlm",
                }
            ],
            "seed_candidates": [
                {
                    "target_class": "SMD",
                    "view_name": "top_view",
                    "template_id": "optional-existing-template-id",
                    "polygon_pct": [[0.1, 0.1], [0.12, 0.1], [0.12, 0.12]],
                    "confidence": 0.72,
                    "source": "vlm",
                }
            ],
            "plugins": ["composite_shape_signature"],
            "settings": {"score_max": 0.4},
        },
        "notes": [
            "When seed_candidates is empty, /agent/run returns a review payload for ROI and seed selection.",
            "Seed candidates may reference an existing template_id, inline entities/group_center, click_pct, or polygon_pct.",
            "When seed_candidates is present, /agent/run resolves each seed and scans with /scan_fast.",
            "The model should output coordinates in render percentage space, not screen pixels.",
        ],
    }


@app.get("/agent/config")
async def get_agent_config():
    config = _vllm_config()
    return {
        "vllm_enabled": config["enabled"],
        "base_url_configured": bool(config["base_url"]),
        "api_key_configured": config["api_key_configured"],
        "llm_model": config["llm_model"] or None,
        "vlm_model": config["vlm_model"] or None,
        "timeout_seconds": config["timeout_seconds"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "send_image": config["send_image"],
        "source": config["source"],
    }


@app.get("/agent/render/{cache_id}")
async def get_agent_render(
    cache_id: str,
    include_svg: bool = Query(False),
):
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )
    svg = cache_payload.get("svg") or ""
    payload = {
        "cache_id": cache_id,
        "bounds": cache_payload.get("bounds"),
        "entity_count": len(cache_payload.get("fingerprints") or []),
        "has_svg": bool(svg),
        "svg_length": len(svg),
        "coord_basis": (
            "render_matrix" if cache_payload.get("render_mapping") else "dxf_extents"
        ),
    }
    if include_svg:
        payload["svg"] = svg
    return payload


@app.get("/agent/regions/{cache_id}")
async def get_agent_regions(
    cache_id: str,
    grid_size: int = Query(VIEW_GRID_SIZE_DEFAULT),
    min_region_cells: int = Query(VIEW_MIN_REGION_CELLS_DEFAULT),
    close_kernel: int = Query(VIEW_CLOSE_KERNEL_DEFAULT),
    connectivity: int = Query(VIEW_CONNECTIVITY_DEFAULT),
    density_quantile: float = Query(VIEW_DENSITY_QUANTILE_DEFAULT),
    include_grid: bool = Query(False),
):
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )
    regions, density = _detect_regions(
        cache_payload,
        settings={
            "view_grid_size": grid_size,
            "view_min_region_cells": min_region_cells,
            "view_close_kernel": close_kernel,
            "view_connectivity": connectivity,
            "view_density_quantile": density_quantile,
        },
    )
    payload = {
        "cache_id": cache_id,
        "bounds": cache_payload.get("bounds"),
        "grid_size": density["gx"],
        "regions": regions,
        "summary": {
            "total_regions": len(regions),
            "view_regions": sum(1 for r in regions if r["label"] == "view"),
            "table_regions": sum(1 for r in regions if r["label"] == "table"),
            "title_block_regions": sum(
                1 for r in regions if r["label"] == "title_block"
            ),
            "dimension_regions": sum(
                1 for r in regions if r["label"] == "dimension"
            ),
            "entity_total": int(density["grid"].sum()),
        },
    }
    if include_grid:
        payload["density_grid"] = density["grid"].tolist()
    return payload


@app.get("/agent/tile/{cache_id}")
async def get_agent_tile(
    cache_id: str,
    x0: float = Query(0.0),
    y0: float = Query(0.0),
    x1: float = Query(1.0),
    y1: float = Query(1.0),
    px: int = Query(TILE_PX_LONG_SIDE_DEFAULT),
):
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )
    roi_pct = _normalize_roi_pct([[x0, y0], [x1, y1]])
    if roi_pct is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid tile rectangle."},
        )
    png_bytes = _render_tile_png(
        cache_payload, roi_pct, px_long_side=max(64, min(int(px), 4096))
    )
    from fastapi.responses import Response

    return Response(content=png_bytes, media_type="image/png")


def _region_to_view(region):
    return {
        "name": region.get("view_name") or region["id"],
        "roi_pct": region["roi_pct"],
        "confidence": region.get("label_confidence"),
        "source": region.get("label_source", "heuristic"),
        "needs_review": True,
        "label": region["label"],
        "region_id": region["id"],
    }


@app.post("/agent/propose")
async def propose_agent_inputs(request: AgentProposeRequest):
    cache_payload = _get_cache(request.cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )
    config = _vllm_config(request.provider)
    if not config["enabled"] or not (config["llm_model"] or config["vlm_model"]):
        return JSONResponse(
            status_code=503,
            content={
                "error": "vLLM is not configured.",
                "provider": _provider_public_debug(config),
                "required": {
                    "request_provider": [
                        "provider.base_url",
                        "provider.vlm_model or provider.llm_model",
                    ],
                    "environment": [
                        "VLLM_BASE_URL",
                        "VLLM_VLM_MODEL or VLLM_LLM_MODEL",
                    ],
                },
                "fallback": {
                    "status": "needs_review",
                    "views": _normalize_agent_views(request.views),
                    "seed_candidates": [],
                },
            },
        )
    if not cache_payload.get("svg"):
        return JSONResponse(
            status_code=400,
            content={"error": "This cache does not include an SVG render artifact."},
        )

    settings = request.settings or {}
    notes = []
    try:
        regions, density = _detect_regions(
            cache_payload,
            instruction=request.instruction,
            target_classes=request.target_classes,
            target_descriptions=request.target_descriptions,
            provider=request.provider,
            settings=settings,
            raise_on_llm_error=bool(config["llm_model"]),
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": str(exc),
                "stage": "region_classification",
                "provider": _provider_public_debug(config),
            },
        )

    view_regions = [r for r in regions if r["label"] in REGION_PIPELINE_LABELS]
    views_payload = [_region_to_view(r) for r in view_regions]

    seed_candidates = []
    tile_debug = []
    seed_errors = []
    if config["vlm_model"] and view_regions:
        for region in view_regions:
            seeds, debug, errors = _propose_seeds_for_region(
                cache_payload,
                region,
                instruction=request.instruction,
                target_classes=request.target_classes or ["SMD"],
                target_descriptions=request.target_descriptions,
                provider=request.provider,
                settings=settings,
            )
            seed_candidates.extend(seeds)
            tile_debug.extend(debug)
            seed_errors.extend(errors)
        if seed_errors and not seed_candidates:
            return JSONResponse(
                status_code=502,
                content={
                    "error": seed_errors[0]["error"],
                    "stage": "seed_proposal",
                    "tile_errors": seed_errors,
                    "provider": _provider_public_debug(config),
                },
            )
        if seed_errors:
            notes.append(
                f"{len(seed_errors)} of {len(tile_debug)} tile(s) failed seed proposal"
            )
    elif not config["vlm_model"]:
        notes.append("No vlm_model configured; seed proposal skipped.")

    proposal = {
        "views": views_payload,
        "seed_candidates": seed_candidates,
        "regions": regions,
        "tile_debug": tile_debug,
        "notes": notes,
    }
    return {
        "status": "proposal_ready",
        "review_required": True,
        "cache_id": request.cache_id,
        "instruction": request.instruction,
        "target_classes": request.target_classes or ["SMD"],
        "provider": {**_provider_public_debug(config)},
        "proposal": proposal,
    }


@app.post("/agent/run")
async def run_agent_pipeline(request: AgentRunRequest):
    cache_payload = _get_cache(request.cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid or expired cache_id. Please upload a file again."
            },
        )

    views = _normalize_agent_views(request.views)
    view_lookup = _agent_view_by_name(views)
    target_classes = request.target_classes or ["SMD"]
    runtime_config = _coerce_runtime_config(request.settings)
    enabled_plugins = sorted(_resolve_enabled_plugins(request.plugins))

    if not request.seed_candidates:
        return {
            "status": "needs_review",
            "review_required": True,
            "next_action": "confirm_view_rois_and_add_seed_candidates",
            "cache_id": request.cache_id,
            "instruction": request.instruction,
            "target_classes": target_classes,
            "views": views,
            "class_tasks": [
                {
                    "target_class": target_class,
                    "required_review": "Pick one or more representative seed templates inside each accepted view ROI.",
                }
                for target_class in target_classes
            ],
            "settings": runtime_config,
            "plugins": enabled_plugins,
        }

    class_results = []
    for seed in request.seed_candidates:
        seed_payload = _agent_seed_payload(seed)
        view = view_lookup.get(seed.view_name) if seed.view_name else None
        roi_pct = view.get("roi_pct") if view else None
        extract_payload = None
        extract_result = None

        if seed.template_id:
            stored_template = _load_extracted_template(request.cache_id, seed.template_id)
            if stored_template is None:
                class_results.append(
                    {
                        "target_class": seed.target_class,
                        "view_name": seed.view_name,
                        "seed": seed_payload,
                        "status": "needs_review",
                        "error": "Seed template_id was not found in the current cache.",
                    }
                )
                continue
            extract_result = {
                "cache_id": request.cache_id,
                "template_id": seed.template_id,
                "selector_mode": "template_id",
                "group_center": stored_template.get("group_center"),
                "entity_count": len(stored_template.get("entities", [])),
                "entities_preview": stored_template.get("entities", [])[
                    : int(runtime_config["extract_entities_preview_limit"])
                ],
                "highlights": [],
                "highlight_labels": [],
            }
        elif seed.entities:
            entities_clean = _sanitize_template_entities(seed.entities)
            group_center = _sanitize_template_group_center(seed.group_center)
            if group_center is None and entities_clean:
                group_center = {
                    "x": round(
                        sum(entity["x"] for entity in entities_clean)
                        / len(entities_clean),
                        3,
                    ),
                    "y": round(
                        sum(entity["y"] for entity in entities_clean)
                        / len(entities_clean),
                        3,
                    ),
                }
            extract_result = {
                "cache_id": request.cache_id,
                "template_id": None,
                "selector_mode": "inline_entities",
                "group_center": group_center,
                "entity_count": len(entities_clean),
                "entities_preview": entities_clean[
                    : int(runtime_config["extract_entities_preview_limit"])
                ],
                "entities": entities_clean,
                "highlights": [],
                "highlight_labels": [],
            }
        elif seed.click_pct:
            extract_payload = {
                "cache_id": request.cache_id,
                "click_pct": seed.click_pct,
                "settings": runtime_config,
            }
            entities_found = _extract_entities_from_click(cache_payload, seed.click_pct)
            extract_result = _build_extract_response(
                request.cache_id,
                cache_payload["bounds"],
                entities_found,
                "click",
                runtime_config=runtime_config,
                render_mapping=cache_payload.get("render_mapping"),
            )
        elif seed.polygon_pct:
            extract_payload = {
                "cache_id": request.cache_id,
                "polygon_pct": seed.polygon_pct,
                "settings": runtime_config,
            }
            entities_found = _extract_entities_from_polygon(
                cache_payload, seed.polygon_pct
            )
            extract_result = _build_extract_response(
                request.cache_id,
                cache_payload["bounds"],
                entities_found,
                "polygon",
                runtime_config=runtime_config,
                render_mapping=cache_payload.get("render_mapping"),
            )
        else:
            class_results.append(
                {
                    "target_class": seed.target_class,
                    "view_name": seed.view_name,
                    "seed": seed_payload,
                    "status": "rejected",
                    "error": "Seed candidate must include template_id, entities, click_pct, or polygon_pct.",
                }
            )
            continue

        if not extract_result.get("template_id") and not extract_result.get("entities"):
            class_results.append(
                {
                    "target_class": seed.target_class,
                    "view_name": seed.view_name,
                    "seed": seed_payload,
                    "status": "needs_review",
                    "extract_request": extract_payload,
                    "extract": extract_result,
                    "error": "Seed did not extract a usable template.",
                }
            )
            continue

        scan_payload = {
            "cache_id": request.cache_id,
            "plugins": enabled_plugins,
            "settings": runtime_config,
        }
        if extract_result.get("template_id"):
            scan_payload["template_id"] = extract_result["template_id"]
        else:
            scan_payload["entities"] = extract_result.get("entities", [])
            scan_payload["group_center"] = extract_result.get("group_center")
        if roi_pct:
            scan_payload["roi_pct"] = roi_pct
        scan_result = _json_response_payload(await scan_dxf_fast(scan_payload))
        validation = _agent_validation_summary(
            extract_result, scan_result, seed_confidence=seed.confidence
        )

        class_results.append(
            {
                "target_class": seed.target_class,
                "view_name": seed.view_name,
                "seed": seed_payload,
                "status": "completed",
                "requires_human_validation": True,
                "validation": validation,
                "extract_request": extract_payload,
                "scan_request": {
                    k: v
                    for k, v in scan_payload.items()
                    if k not in {"entities"}
                },
                "extract": {
                    "template_id": extract_result.get("template_id"),
                    "entity_count": extract_result.get("entity_count", 0),
                    "group_center": extract_result.get("group_center"),
                    "highlights": extract_result.get("highlights", []),
                    "highlight_labels": extract_result.get("highlight_labels", []),
                },
                "scan": scan_result,
            }
        )

    return {
        "status": "completed_with_review",
        "review_required": True,
        "cache_id": request.cache_id,
        "instruction": request.instruction,
        "target_classes": target_classes,
        "views": views,
        "results": class_results,
        "matches_by_class": _group_agent_results(class_results),
        "settings": runtime_config,
        "plugins": enabled_plugins,
    }


@app.get("/template_library")
async def get_template_library():
    return _template_library_json_response(_sqlite_library_response())


@app.post("/template_library/create_library")
async def create_template_library(data: dict = Body(...)):
    label = str(data.get("label") or "").strip()
    if not label:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing library label."},
        )
    created = _sqlite_create_library(label)
    return _template_library_json_response(
        {"created_library": created, "library": _sqlite_library_response()}
    )


@app.post("/template_library/rename_library")
async def rename_template_library(data: dict = Body(...)):
    library_id = str(data.get("library_id") or data.get("version_id") or "").strip()
    label = str(data.get("label") or "").strip()
    if not library_id or not label:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing library_id or label."},
        )
    if not _sqlite_rename_library(library_id, label):
        return JSONResponse(
            status_code=404,
            content={"error": "Template library not found."},
        )
    return _template_library_json_response(_sqlite_library_response())


@app.post("/template_library/select_library")
async def select_template_library(data: dict = Body(...)):
    library_id = str(data.get("library_id") or data.get("version_id") or "").strip()
    if not library_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing library_id."},
        )
    if not _sqlite_select_library(library_id):
        return JSONResponse(
            status_code=404,
            content={"error": "Template library not found."},
        )
    return _template_library_json_response(_sqlite_library_response())


@app.post("/template_library/select_version")
async def select_template_library_version(data: dict = Body(...)):
    version_id = str(data.get("version_id") or data.get("library_id") or "").strip()
    if not version_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing library_id."},
        )
    if not _sqlite_select_library(version_id):
        return JSONResponse(
            status_code=404,
            content={"error": "Template library not found."},
        )
    return _template_library_json_response(_sqlite_library_response())


@app.post("/template_library/delete_library")
async def delete_template_library(data: dict = Body(...)):
    library_id = str(data.get("library_id") or data.get("version_id") or "").strip()
    if not library_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing library_id."},
        )
    if not _sqlite_delete_library(library_id):
        return JSONResponse(
            status_code=404,
            content={"error": "Template library not found."},
        )
    return _template_library_json_response(_sqlite_library_response())


@app.post("/template_library/delete_version")
async def delete_template_library_version(data: dict = Body(...)):
    version_id = str(data.get("version_id") or data.get("library_id") or "").strip()
    if not version_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing library_id."},
        )
    if not _sqlite_delete_library(version_id):
        return JSONResponse(
            status_code=404,
            content={"error": "Template library not found."},
        )
    return _template_library_json_response(_sqlite_library_response())


@app.post("/template_library/save")
async def save_template_library(data: dict = Body(...)):
    cache_id = data.get("cache_id")
    template_id = data.get("template_id")
    if template_id:
        template = _load_extracted_template(cache_id, template_id)
        if template is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found. Please extract template again."},
            )
        group_center = template.get("group_center")
        entities = template.get("entities", [])
    else:
        group_center = data.get("group_center")
        entities = data.get("entities", [])

    saved = _sqlite_save_template_library_entry(
        data.get("name"),
        data.get("category"),
        group_center,
        entities,
    )
    if saved is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid template entities found to save."},
        )

    return _template_library_json_response(
        {
            "saved_template": _serialize_saved_template(saved),
            "library": _sqlite_library_response(),
        }
    )


@app.post("/template_library/delete_template")
async def delete_template_library_template(data: dict = Body(...)):
    template_id = str(data.get("template_id") or "").strip()
    if not template_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing template_id."},
        )
    if not _sqlite_delete_template(template_id):
        return JSONResponse(
            status_code=404,
            content={"error": "Template not found in active library."},
        )
    return _template_library_json_response(_sqlite_library_response())


@app.post("/template_library/delete_category")
async def delete_template_library_category(data: dict = Body(...)):
    category = str(data.get("category") or "").strip()
    if not category:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing category."},
        )
    deleted_count = _sqlite_delete_category(category)
    if deleted_count == 0:
        return JSONResponse(
            status_code=404,
            content={"error": "Category not found in active library."},
        )
    return _template_library_json_response(
        {
            "deleted_category": _normalize_template_category(category),
            "deleted_count": deleted_count,
            "library": _sqlite_library_response(),
        }
    )


@app.post("/template_library/resolve")
async def resolve_template_library(data: dict = Body(...)):
    records = _sqlite_resolve_template_entries(data.get("template_ids", []))
    return {
        "templates": [
            _serialize_saved_template(record, include_entities=True)
            for record in records
        ]
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
    render_mapping = None
    try:
        matrix = backend.transformation_matrix
        inv = matrix.copy()
        inv.inverse()
        root = ET.fromstring(svg)
        vb = (root.get("viewBox") or "").strip().split()
        if len(vb) == 4:
            viewbox_x = float(vb[0])
            viewbox_y = float(vb[1])
            viewbox_w = float(vb[2])
            viewbox_h = float(vb[3])
            if viewbox_w > 0 and viewbox_h > 0:
                render_mapping = {
                    "matrix": [float(v) for row in matrix.rows() for v in row],
                    "inv_matrix": [float(v) for row in inv.rows() for v in row],
                    "viewbox_x": viewbox_x,
                    "viewbox_y": viewbox_y,
                    "viewbox_w": viewbox_w,
                    "viewbox_h": viewbox_h,
                }
    except Exception:
        render_mapping = None
    svg_render_time_ms = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    cache_payload = _build_cache_payload(
        msp,
        fast_build=fast_build,
        render_mapping=render_mapping,
    )
    cache_payload["svg"] = svg
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
        "coord_basis": ("render_matrix" if render_mapping else "dxf_extents"),
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
    render_mapping = cache_payload.get("render_mapping")
    if isinstance(click_pct, list) and len(click_pct) >= 2:
        entities_found = _extract_entities_from_click(cache_payload, click_pct)
        return _build_extract_response(
            cache_id,
            bounds,
            entities_found,
            "click",
            runtime_config=runtime_config,
            render_mapping=render_mapping,
        )

    entities_found = _extract_entities_from_polygon(cache_payload, polygon_pct)
    return _build_extract_response(
        cache_id,
        bounds,
        entities_found,
        "polygon",
        runtime_config=runtime_config,
        render_mapping=render_mapping,
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
            geometry_to_render_pct(
                f["geometry"], payload["bounds"], payload.get("render_mapping")
            )
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
    render_mapping=None,
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

    rpx, rpy = _dxf_point_to_render_pct((mcx, mcy), bounds, render_mapping)

    mf = [af] + [all_fp[i] for i in used]
    handle_ids = _unique_handle_ids(
        handle_id for feature in mf for handle_id in feature.get("handleIDs", [])
    )
    selected = mf if max_highlights is None else mf[:max_highlights]
    hl = [
        _compact_render_highlight(
            geometry_to_render_pct(f["geometry"], bounds, render_mapping)
        )
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
        "handleIDs": handle_ids,
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


def _roi_box_from_pct(cache_payload, roi_pct):
    if not isinstance(roi_pct, list) or len(roi_pct) != 2:
        return None
    if not all(isinstance(pt, list) and len(pt) >= 2 for pt in roi_pct):
        return None
    bounds = cache_payload["bounds"]
    render_mapping = cache_payload.get("render_mapping")
    try:
        p1 = _render_pct_to_dxf_point(roi_pct[0], bounds, render_mapping)
        p2 = _render_pct_to_dxf_point(roi_pct[1], bounds, render_mapping)
    except (TypeError, ValueError):
        return None
    dxf_min_x = min(p1[0], p2[0])
    dxf_max_x = max(p1[0], p2[0])
    dxf_min_y = min(p1[1], p2[1])
    dxf_max_y = max(p1[1], p2[1])
    if dxf_max_x - dxf_min_x <= BOUNDS_EPS or dxf_max_y - dxf_min_y <= BOUNDS_EPS:
        return None
    return ShapelyPolygon(
        [
            (dxf_min_x, dxf_min_y),
            (dxf_max_x, dxf_min_y),
            (dxf_max_x, dxf_max_y),
            (dxf_min_x, dxf_max_y),
        ]
    )


def _feature_within_roi(feature, roi_box):
    geom = feature.get("geometry") or {}
    kind = geom.get("kind")
    if kind == "circle":
        try:
            cx = float(geom["cx"])
            cy = float(geom["cy"])
            r = abs(float(geom["r"]))
        except (KeyError, TypeError, ValueError):
            return roi_box.covers(Point(feature["x"], feature["y"]))
        return roi_box.covers(
            ShapelyPolygon(
                [
                    (cx - r, cy - r),
                    (cx + r, cy - r),
                    (cx + r, cy + r),
                    (cx - r, cy + r),
                ]
            )
        )
    if kind == "polyline":
        try:
            points = geom.get("points", [])
            if len(points) >= 2:
                return roi_box.covers(LineString(points))
        except Exception:
            pass
    return roi_box.covers(Point(feature["x"], feature["y"]))


def _apply_roi_filter(cache_payload, roi_pct):
    """Filter fingerprints to ROI rectangle. roi_pct=[[x1,y1],[x2,y2]] in render-pct space."""
    roi_box = _roi_box_from_pct(cache_payload, roi_pct)
    if roi_box is None:
        filtered_fp = []
    else:
        filtered_fp = [
            f for f in cache_payload["fingerprints"] if _feature_within_roi(f, roi_box)
        ]
    if filtered_fp:
        coords = np.array([[f["x"], f["y"]] for f in filtered_fp])
        sizes = np.array([f["size"] for f in filtered_fp])
        tree = cKDTree(coords)
    else:
        coords = np.empty((0, 2))
        sizes = np.empty(0)
        tree = None
    type_index = {}
    for idx, f in enumerate(filtered_fp):
        type_index.setdefault(f["type"], []).append(idx)
    type_index = {t: np.array(v) for t, v in type_index.items()}
    return filtered_fp, tree, coords, sizes, type_index


def _highlight_within_roi_pct(highlight, roi_pct):
    if not isinstance(roi_pct, list) or len(roi_pct) != 2:
        return True
    try:
        x1, y1 = float(roi_pct[0][0]), float(roi_pct[0][1])
        x2, y2 = float(roi_pct[1][0]), float(roi_pct[1][1])
    except (TypeError, ValueError, IndexError):
        return True
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    kind = highlight.get("kind")
    if kind == "circle":
        try:
            cx = float(highlight["cx_pct"])
            cy = float(highlight["cy_pct"])
            r = abs(float(highlight.get("r_pct", 0.0)))
        except (TypeError, ValueError, KeyError):
            return False
        return (
            min_x <= cx - r and cx + r <= max_x and min_y <= cy - r and cy + r <= max_y
        )
    if kind == "polyline":
        points = highlight.get("points_pct", [])
        if not points:
            return False
        try:
            return all(
                min_x <= float(pt[0]) <= max_x and min_y <= float(pt[1]) <= max_y
                for pt in points
            )
        except (TypeError, ValueError, IndexError):
            return False
    return False


def _match_within_roi_pct(match, roi_pct):
    highlights = match.get("highlights") or []
    if not highlights:
        return FalsGe
    return all(_highlight_within_roi_pct(h, roi_pct) for h in highlights)


def _filter_matches_to_roi_pct(matches, roi_pct):
    if not roi_pct:
        return matches
    return [m for m in matches if _match_within_roi_pct(m, roi_pct)]


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
    render_mapping = cache_payload.get("render_mapping")
    all_fp = cache_payload["fingerprints"]
    tree = cache_payload["tree"]
    type_index = cache_payload["type_index"]
    sizes = cache_payload["sizes"]

    roi_pct = template.get("roi_pct")
    if roi_pct and len(roi_pct) == 2:
        all_fp, tree, _, sizes, type_index = _apply_roi_filter(cache_payload, roi_pct)
        stats["roi_fp_count"] = len(all_fp)
        stats["total_fp_count"] = len(cache_payload["fingerprints"])
        stats["roi_pct"] = roi_pct

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
                    render_mapping=render_mapping,
                )
            )
        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
        stats["raw_match_count"] = len(matches_found)
        unique = _dedupe_and_sort_matches(matches_found)
        if roi_pct:
            before_roi_match_count = len(unique)
            unique = _filter_matches_to_roi_pct(unique, roi_pct)
            stats["roi_match_reject_count"] = before_roi_match_count - len(unique)
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
                    render_mapping=render_mapping,
                )
            )
        else:
            stats["matching_fail_count"] += 1

    stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
    stats["raw_match_count"] = len(matches_found)
    unique = _dedupe_and_sort_matches(matches_found)
    if roi_pct:
        before_roi_match_count = len(unique)
        unique = _filter_matches_to_roi_pct(unique, roi_pct)
        stats["roi_match_reject_count"] = before_roi_match_count - len(unique)
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
    render_mapping = cache_payload.get("render_mapping")
    all_fp = cache_payload["fingerprints"]
    coords = cache_payload["coords"]
    sizes = cache_payload["sizes"]
    tree = cache_payload["tree"]
    type_index = cache_payload["type_index"]

    roi_pct = template.get("roi_pct")
    if roi_pct and len(roi_pct) == 2:
        all_fp, tree, coords, sizes, type_index = _apply_roi_filter(
            cache_payload, roi_pct
        )
        stats["roi_fp_count"] = len(all_fp)
        stats["total_fp_count"] = len(cache_payload["fingerprints"])
        stats["roi_pct"] = roi_pct

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
                    render_mapping=render_mapping,
                )
            )
        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
        stats["raw_match_count"] = len(matches_found)
        unique = _dedupe_and_sort_matches(matches_found)
        if roi_pct:
            before_roi_match_count = len(unique)
            unique = _filter_matches_to_roi_pct(unique, roi_pct)
            stats["roi_match_reject_count"] = before_roi_match_count - len(unique)
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
                    render_mapping=render_mapping,
                )
            )
        else:
            stats["matching_fail_count"] += 1

    stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
    stats["raw_match_count"] = len(matches_found)
    unique = _dedupe_and_sort_matches(matches_found)
    if roi_pct:
        before_roi_match_count = len(unique)
        unique = _filter_matches_to_roi_pct(unique, roi_pct)
        stats["roi_match_reject_count"] = before_roi_match_count - len(unique)
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

    uvicorn.run(app, host="0.0.0.0", port=8008)
