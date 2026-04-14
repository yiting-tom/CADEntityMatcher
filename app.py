import math
import os
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
import json
from collections import Counter
from typing import Callable
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, UploadFile, File, Body, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import ezdxf
from ezdxf.bbox import extents
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.math import Matrix44
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely.ops import linemerge

# Initialize FastAPI app
app = FastAPI()

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
MATCH_SCORE_MAX_DEFAULT = 0.40
MATCH_SCORE_ANCHOR_WEIGHT = 0.30
MATCH_SCORE_SIZE_WEIGHT = 0.45
MATCH_SCORE_DIST_WEIGHT = 0.45
MATCH_SCORE_SHAPE_WEIGHT = 0.10
BOUNDS_EPS = 1e-9
MAX_INSERT_EXPLODE_DEPTH = 8
CIRCLE_MODE_RADIUS_DECIMALS = 6
CIRCLE_MODE_MIN_GROUP_COUNT = 2
DEFAULT_DROP_ENTITY_TYPES = {
    "TEXT",
    "MTEXT",
    "DIMENSION",
    "LEADER",
    "MLEADER",
    "HATCH",
    "IMAGE",
    "WIPEOUT",
    "POINT",
    "XLINE",
    "RAY",
}


def snap(val):
    """Round coordinates to fixed precision for stable linemerge endpoints."""
    return round(val, SNAP_DECIMALS)


# --- Pydantic models (normalized geometry fingerprint) ---
class EntityModel(BaseModel):
    type: str  # CIRCLE, COMPOSITE_SHAPE
    size: float  # Radius or total length/perimeter
    x: float  # Geometry center X
    y: float  # Geometry center Y


class TemplateModel(BaseModel):
    entities: list[EntityModel]


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


def _bbox2d_to_bounds(bbox2d):
    """Convert ezdxf BoundingBox2d to internal bounds dict."""
    min_x, min_y = bbox2d.extmin
    max_x, max_y = bbox2d.extmax
    return _make_bounds(min_x, min_y, max_x, max_y)


def get_dxf_bounds(msp):
    """Get DXF extents and dimensions."""
    ext = extents(msp)
    return _make_bounds(ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y)


def _iter_flat_entities(entities, *, depth=0, max_depth=MAX_INSERT_EXPLODE_DEPTH):
    """Yield entities with INSERT exploded to virtual entities recursively."""
    if depth > max_depth:
        return
    for entity in entities:
        if entity.dxftype() == "INSERT":
            if depth >= max_depth:
                continue
            try:
                yield from _iter_flat_entities(
                    entity.virtual_entities(), depth=depth + 1, max_depth=max_depth
                )
            except Exception:
                continue
            continue
        yield entity


def _sanitize_insert_entity(insert):
    """Fix degenerate INSERT transform attributes in-place."""
    changed = False
    for attr in ("xscale", "yscale", "zscale"):
        if not insert.dxf.hasattr(attr):
            continue
        try:
            v = float(getattr(insert.dxf, attr))
        except Exception:
            v = 1.0
        if not math.isfinite(v) or abs(v) < BOUNDS_EPS:
            setattr(insert.dxf, attr, 1.0)
            changed = True

    if insert.dxf.hasattr("rotation"):
        try:
            rot = float(insert.dxf.rotation)
        except Exception:
            rot = 0.0
        if not math.isfinite(rot):
            insert.dxf.rotation = 0.0
            changed = True

    if insert.dxf.hasattr("extrusion"):
        try:
            ex = insert.dxf.extrusion
            mag = math.sqrt(ex.x * ex.x + ex.y * ex.y + ex.z * ex.z)
        except Exception:
            mag = 0.0
        if mag < BOUNDS_EPS or not math.isfinite(mag):
            insert.dxf.extrusion = (0.0, 0.0, 1.0)
            changed = True
    return changed


def _iter_all_entity_spaces(doc):
    """Yield modelspace and all block entity spaces."""
    yield doc.modelspace()
    for blk in doc.blocks:
        yield blk


def _sorted_type_counts(counter):
    """Sort type counts by count desc then name asc."""
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _count_entity_types(entities):
    """Count DXF entity types from an iterable of entities."""
    c = Counter()
    for e in entities:
        try:
            c[e.dxftype().upper()] += 1
        except Exception:
            continue
    return _sorted_type_counts(c)


def _drop_entities_by_type(doc, type_names):
    """Drop entities by DXF type from modelspace and blocks."""
    targets = {str(t).upper() for t in type_names}
    removed = 0
    removed_by_type = {}
    for space in _iter_all_entity_spaces(doc):
        for entity in list(space):
            etype = entity.dxftype().upper()
            if etype not in targets:
                continue
            try:
                space.delete_entity(entity)
                removed += 1
                removed_by_type[etype] = removed_by_type.get(etype, 0) + 1
            except Exception:
                pass
    return removed, removed_by_type


def _parse_entity_types_form(raw):
    """Parse entity type list from form input (JSON array or CSV string)."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []

    values = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            values = [str(v) for v in parsed]
        elif isinstance(parsed, str):
            values = [parsed]
    except Exception:
        values = text.replace(";", ",").split(",")

    out = []
    seen = set()
    for v in values:
        t = str(v).strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _sanitize_doc_inserts(doc):
    """Sanitize INSERT attributes across modelspace and blocks."""
    scanned = 0
    fixed = 0
    for space in _iter_all_entity_spaces(doc):
        for entity in space:
            if entity.dxftype() != "INSERT":
                continue
            scanned += 1
            if _sanitize_insert_entity(entity):
                fixed += 1
    return scanned, fixed


def _drop_unrenderable_inserts(doc):
    """Remove INSERT entities that still fail virtual expansion."""
    removed = 0
    for space in _iter_all_entity_spaces(doc):
        for entity in list(space):
            if entity.dxftype() != "INSERT":
                continue
            try:
                # Force virtual transformation path used by renderer.
                for _ in entity.virtual_entities():
                    break
            except Exception:
                try:
                    space.delete_entity(entity)
                    removed += 1
                except Exception:
                    pass
    return removed


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
    3. Expands INSERT (block references) recursively via virtual_entities().
    4. Snap endpoints to improve linemerge stability.
    5. Apply ROI filtering last (centroid must be inside ROI).
    """
    circles = []
    all_lines = []

    for entity in _iter_flat_entities(msp):
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
                LineString(
                    [(snap(p1.x), snap(p1.y)), (snap(p2.x), snap(p2.y))]
                )
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
                pts = [
                    (snap(p.x), snap(p.y)) for p in entity.flattening(flatten_tol)
                ]
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


def _build_cache_payload(
    msp, *, fast_build=False, bounds_override=None, render_mapping=None
):
    """Build cache payload (fingerprints + spatial indexes) for one upload."""
    bounds = bounds_override or get_dxf_bounds(msp)
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


def _pick_anchor(entities, all_fp, type_index, sizes):
    """Pick the entity with the fewest global candidates as anchor (rarest first)."""
    best_idx, best_cnt = 0, float("inf")
    for i, e in enumerate(entities):
        t = e["type"]
        if t not in type_index:
            return i  # Type not found globally -> 0 candidates; fast fail
        tol = max(SIZE_TOL_MIN, e["size"] * SIZE_TOL_RATIO)
        base_idx = type_index[t][np.abs(sizes[type_index[t]] - e["size"]) < tol]
        if len(base_idx) == 0:
            return i
        cnt = sum(1 for idx in base_idx if _single_entity_plugin_pass(e, all_fp[int(idx)]))
        if cnt < best_cnt:
            best_cnt = cnt
            best_idx = i
    return best_idx


SingleEntityPlugin = Callable[[dict, dict], bool]
SINGLE_ENTITY_MATCH_PLUGINS: list[SingleEntityPlugin] = []

UploadDocPlugin = Callable[[object, dict], dict]
UPLOAD_DOC_PLUGINS: dict[str, UploadDocPlugin] = {}


def register_single_entity_plugin(plugin: SingleEntityPlugin):
    """Register a single-entity matching plugin."""
    if plugin not in SINGLE_ENTITY_MATCH_PLUGINS:
        SINGLE_ENTITY_MATCH_PLUGINS.append(plugin)


def register_upload_doc_plugin(name: str, plugin: UploadDocPlugin):
    """Register a document pre-process plugin used by /upload."""
    key = str(name).strip()
    if key:
        UPLOAD_DOC_PLUGINS[key] = plugin


def _run_upload_doc_plugin(name: str, doc, options=None):
    """Run one upload plugin by name."""
    plugin = UPLOAD_DOC_PLUGINS.get(str(name))
    if plugin is None:
        return {"enabled": False, "applied": False, "error": f"Unknown plugin: {name}"}
    try:
        result = plugin(doc, options or {})
        if isinstance(result, dict):
            return result
        return {"enabled": True, "applied": False, "error": "Invalid plugin result"}
    except Exception as exc:
        return {"enabled": True, "applied": False, "error": str(exc)}


def _plugin_drop_most_common_circle_size(doc, options):
    """Drop all CIRCLE entities in the most frequent radius bucket."""
    decimals = int(options.get("decimals", CIRCLE_MODE_RADIUS_DECIMALS))
    min_group_count = int(options.get("min_group_count", CIRCLE_MODE_MIN_GROUP_COUNT))
    buckets = {}

    msp = doc.modelspace()
    for entity in list(msp):
        if entity.dxftype() != "CIRCLE":
            continue
        try:
            radius = float(entity.dxf.radius)
        except Exception:
            continue
        if not math.isfinite(radius) or radius <= BOUNDS_EPS:
            continue
        key = round(radius, decimals)
        buckets.setdefault(key, []).append((msp, entity))

    if not buckets:
        return {
            "enabled": True,
            "applied": False,
            "removed_count": 0,
            "radius": None,
            "bucket_count": 0,
            "unique_radius_count": 0,
        }

    best_radius, best_bucket = sorted(
        buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])
    )[0]
    if len(best_bucket) < max(1, min_group_count):
        return {
            "enabled": True,
            "applied": False,
            "removed_count": 0,
            "radius": best_radius,
            "bucket_count": len(best_bucket),
            "unique_radius_count": len(buckets),
        }

    removed = 0
    for space, entity in best_bucket:
        try:
            space.delete_entity(entity)
            removed += 1
        except Exception:
            pass

    return {
        "enabled": True,
        "applied": True,
        "removed_count": int(removed),
        "radius": float(best_radius),
        "bucket_count": len(best_bucket),
        "unique_radius_count": len(buckets),
    }


def _plugin_composite_shape_signature(template_entity, candidate_feature):
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
        if abs(int(tp) - int(cp)) > SINGLE_POINT_COUNT_TOL:
            return False

    ta = ts.get("aspect_ratio")
    ca = cs.get("aspect_ratio")
    if isinstance(ta, (int, float)) and isinstance(ca, (int, float)) and ta > 0:
        if abs(ca - ta) / ta > SINGLE_ASPECT_RATIO_TOL:
            return False

    td = ts.get("bbox_diag")
    cd = cs.get("bbox_diag")
    if isinstance(td, (int, float)) and isinstance(cd, (int, float)) and td > 0:
        if abs(cd - td) / td > SINGLE_DIAG_TOL:
            return False
    return True


def _single_entity_plugin_pass(template_entity, candidate_feature):
    for plugin in SINGLE_ENTITY_MATCH_PLUGINS:
        if not plugin(template_entity, candidate_feature):
            return False
    return True


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def _shape_signature_error(template_entity, candidate_feature):
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
        parts.append(_clamp01(abs(int(tp) - int(cp)) / max(1, SINGLE_POINT_COUNT_TOL)))

    ta = ts.get("aspect_ratio")
    ca = cs.get("aspect_ratio")
    if isinstance(ta, (int, float)) and isinstance(ca, (int, float)) and ta > 0:
        rel = abs(ca - ta) / ta
        parts.append(_clamp01(rel / SINGLE_ASPECT_RATIO_TOL))

    td = ts.get("bbox_diag")
    cd = cs.get("bbox_diag")
    if isinstance(td, (int, float)) and isinstance(cd, (int, float)) and td > 0:
        rel = abs(cd - td) / td
        parts.append(_clamp01(rel / SINGLE_DIAG_TOL))

    if not parts:
        return 0.0
    return float(sum(parts) / len(parts))


def _pair_match_score(
    template_entity,
    candidate_feature,
    *,
    size_tol,
    dist_error_ratio=None,
):
    """Compute normalized pair match score (lower is better)."""
    size_err = _clamp01(abs(candidate_feature["size"] - template_entity["size"]) / size_tol)
    shape_err = _shape_signature_error(template_entity, candidate_feature)
    if dist_error_ratio is None:
        return round(0.85 * size_err + 0.15 * shape_err, 6)
    dist_err = _clamp01(dist_error_ratio)
    return round(
        MATCH_SCORE_SIZE_WEIGHT * size_err
        + MATCH_SCORE_DIST_WEIGHT * dist_err
        + MATCH_SCORE_SHAPE_WEIGHT * shape_err,
        6,
    )


def _resolve_score_max(template):
    raw = template.get("score_max", MATCH_SCORE_MAX_DEFAULT)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return MATCH_SCORE_MAX_DEFAULT
    return _clamp01(v)


def _entity_basic_match(template_entity, candidate_feature, size_tol=None):
    """Base entity matching: type + size (+ optional shape plugins)."""
    if candidate_feature.get("type") != template_entity.get("type"):
        return False
    tol = (
        size_tol
        if size_tol is not None
        else max(SIZE_TOL_MIN, template_entity["size"] * SIZE_TOL_RATIO)
    )
    if abs(candidate_feature.get("size", 0.0) - template_entity["size"]) >= tol:
        return False
    return _single_entity_plugin_pass(template_entity, candidate_feature)


def _single_entity_matches(all_fp, anchor_tmpl):
    """Find single-entity matches by base filters + pluggable plugin filters."""
    return [
        i
        for i, f in enumerate(all_fp)
        if _entity_basic_match(anchor_tmpl, f)
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


register_single_entity_plugin(_plugin_composite_shape_signature)
register_upload_doc_plugin(
    "drop_most_common_circle_size", _plugin_drop_most_common_circle_size
)


# --- HTML frontend ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DXF CAD Pattern Scanner</title>
    <style>
        body { font-family: sans-serif; padding: 20px; max-width: none; margin: 0; }
        .step-container { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
        #canvas-container { position: relative; display: block; width: min(96vw, 1800px); height: clamp(640px, 80vh, 1200px); border: 1px solid #999; margin-top: 10px; cursor: crosshair; background: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; }
        #transform-wrapper { transform-origin: 0 0; position: relative; width: 100%; height: 100%; }
        #svg-display { width: 100%; height: 100%; display: block; }
        #svg-display svg { width: 100%; height: 100%; display: block; }
        button { padding: 8px 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .btn-sm { padding: 4px 10px; font-size: 0.85em; background: #6c757d; }
        .btn-sm:hover { background: #5a6268; }
        pre { background: #eee; padding: 10px; border-radius: 4px; overflow-x: auto; max-height: 250px; }
        .hint { color: #666; font-size: 0.9em; margin: 6px 0; }
        .type-grid { display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 6px; }
        .type-opt { font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>PCB CAD Pattern Scanner</h1>

    <div class="step-container">
        <h3>Step 1: Upload DXF File</h3>
        <input type="file" id="dxf-file" accept=".dxf">
        <label style="margin-left:10px;">
            <input type="checkbox" id="fast-cache-build"> Fast Cache Build (skip ELLIPSE/SPLINE)
        </label>
        <label style="margin-left:10px;">
            <input type="checkbox" id="drop-noisy-types" checked> Enable Entity Type Filtering
        </label>
        <label style="margin-left:10px;">
            <input type="checkbox" id="drop-most-common-circle"> Plugin: Drop Most Common CIRCLE Size
        </label>
        <details style="margin-top:8px;">
            <summary>Choose entity types to drop</summary>
            <div class="type-grid">
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="TEXT" checked> TEXT</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="MTEXT" checked> MTEXT</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="DIMENSION" checked> DIMENSION</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="LEADER" checked> LEADER</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="MLEADER" checked> MLEADER</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="HATCH" checked> HATCH</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="IMAGE" checked> IMAGE</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="WIPEOUT" checked> WIPEOUT</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="POINT" checked> POINT</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="XLINE" checked> XLINE</label>
                <label class="type-opt"><input type="checkbox" class="drop-type-opt" value="RAY" checked> RAY</label>
            </div>
            <div style="margin-top:6px;">
                <input id="drop-type-custom" type="text" style="width:min(100%,560px);" placeholder="Custom types (comma-separated), e.g. ATTDEF,ATTRIB">
            </div>
        </details>
        <button onclick="uploadDXF()">Upload &amp; Render SVG</button>
        <h4>Upload Stats:</h4>
        <pre id="upload-result">No upload yet</pre>
    </div>

    <div class="step-container">
        <h3>Step 2: Select Feature (Polygon)</h3>
        <p class="hint">Left-click to add vertex | Double-click or click green first vertex to close | Right-drag to pan | Scroll to zoom | ESC to clear</p>
        <button class="btn-sm" onclick="clearPoly()">Clear Selection</button>
        <button class="btn-sm" onclick="resetView()">Reset View</button>
        <div id="canvas-container">
            <div id="transform-wrapper">
                <div id="svg-display"></div>
                <svg id="svg-overlay" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;">
                    <g id="poly-group"></g>
                    <g id="template-group"></g>
                    <g id="hl-group"></g>
                </svg>
            </div>
        </div>
        <h4>Extracted Template Fingerprint:</h4>
        <button id="btn-download-template" class="btn-sm" onclick="downloadTemplateFingerprint()" disabled>Download Fingerprint</button>
        <pre id="extract-result">No data yet</pre>
    </div>

    <div class="step-container">
        <h3>Step 3: Full Scan (Analyzer)</h3>
        <label><input type="radio" name="scan-mode" value="/scan" checked> Standard Scan</label>
        <label style="margin-left:12px;"><input type="radio" name="scan-mode" value="/scan_fast"> Fast Scan (KD-Tree)</label>
        <br><br>
        <button id="btn-scan" onclick="scanTemplate()" disabled>Start Pattern Matching</button>
        <button id="btn-download-matches" class="btn-sm" onclick="downloadMatchResults()" disabled>Download Match Results</button>
        <h4 id="scan-status"></h4>
        <pre id="scan-result">Waiting for scan...</pre>
    </div>

    <script>
        // --- DOM ---
        const container = document.getElementById('canvas-container');
        const wrapper   = document.getElementById('transform-wrapper');
        const svgDisplay = document.getElementById('svg-display');
        const svgOverlay = document.getElementById('svg-overlay');
        const uploadResult = document.getElementById('upload-result');
        const polyG = document.getElementById('poly-group');
        const tmplG = document.getElementById('template-group');
        const hlG   = document.getElementById('hl-group');
        const btnDownloadTemplate = document.getElementById('btn-download-template');
        const btnDownloadMatches = document.getElementById('btn-download-matches');

        // --- State ---
        let scale = 1, panX = 0, panY = 0;
        let isPanning = false, panLastX = 0, panLastY = 0;
        let polyPts = [], polyClosed = false;
        let currentTemplate = null;
        let currentMatchResult = null;
        let currentCacheId = null;
        let contentW = 0, contentH = 0;
        let svgViewport = { x0: 0, y0: 0, w: 0, h: 0 };
        const MAX_TEMPLATE_HIGHLIGHTS_TO_DRAW = 250;
        const MAX_MATCHES_TO_DRAW = 200;
        const MAX_HIGHLIGHTS_PER_MATCH_TO_DRAW = 80;
        const MAX_TOTAL_HIGHLIGHTS_TO_DRAW = 1800;
        const MAX_ENTITIES_PREVIEW = 60;
        const MAX_MATCHES_PREVIEW = 80;
        const MAX_POLYLINE_POINTS_DRAW = 400;

        // --- Transform ---
        function applyTransform() {
            wrapper.style.transform = 'translate(' + panX + 'px,' + panY + 'px) scale(' + scale + ')';
        }
        function syncCanvasSize() {
            contentW = container.clientWidth;
            contentH = container.clientHeight;
            svgOverlay.setAttribute("viewBox", "0 0 " + contentW + " " + contentH);
            svgViewport = getSvgViewport();
        }
        function screenToContent(cx, cy) {
            var r = container.getBoundingClientRect();
            return { x: (cx - r.left - panX) / scale, y: (cy - r.top - panY) / scale };
        }
        function getSvgViewport() {
            var fallback = { x0: 0, y0: 0, w: contentW, h: contentH };
            var svgEl = svgDisplay.querySelector('svg');
            if (!svgEl || !contentW || !contentH) return fallback;

            var vb = svgEl.viewBox && svgEl.viewBox.baseVal;
            if (!vb || !vb.width || !vb.height) return fallback;

            var par = (svgEl.getAttribute('preserveAspectRatio') || 'xMidYMid meet').trim();
            par = par.replace(/^defer\\s+/, '');
            if (par === 'none') return fallback;

            var parts = par.split(/\\s+/);
            var align = parts[0] || 'xMidYMid';
            var mode = parts[1] || 'meet';

            var sx = contentW / vb.width;
            var sy = contentH / vb.height;
            var s = mode === 'slice' ? Math.max(sx, sy) : Math.min(sx, sy);
            var vw = vb.width * s;
            var vh = vb.height * s;

            var ax = align.indexOf('xMin') === 0 ? 0 : (align.indexOf('xMax') === 0 ? 1 : 0.5);
            var ay = align.indexOf('YMin') >= 0 ? 0 : (align.indexOf('YMax') >= 0 ? 1 : 0.5);

            return {
                x0: (contentW - vw) * ax,
                y0: (contentH - vh) * ay,
                w: vw,
                h: vh,
            };
        }
        function contentToSvgPct(p) {
            if (!svgViewport.w || !svgViewport.h) return [0, 0];
            return [
                (p.x - svgViewport.x0) / svgViewport.w,
                (p.y - svgViewport.y0) / svgViewport.h,
            ];
        }
        function svgPctToContent(px, py) {
            return {
                x: svgViewport.x0 + px * svgViewport.w,
                y: svgViewport.y0 + py * svgViewport.h,
            };
        }
        function samplePoints(points, maxPoints) {
            if (!points || points.length <= maxPoints) return points || [];
            var step = Math.ceil(points.length / maxPoints);
            var out = [];
            for (var i = 0; i < points.length; i += step) out.push(points[i]);
            var last = points[points.length - 1];
            if (out.length === 0 || out[out.length - 1] !== last) out.push(last);
            return out;
        }
        function setExtractPreview(result) {
            if (!result || result.error) {
                document.getElementById('extract-result').innerText = JSON.stringify(result, null, 2);
                return;
            }
            var entities = result.entities_preview || result.entities || [];
            var highlights = result.highlights || [];
            var entityTotal = result.entity_count ?? entities.length;
            var highlightTotal = result.highlight_count_total ?? highlights.length;
            var preview = {
                cache_id: result.cache_id,
                template_id: result.template_id,
                group_center: result.group_center,
                entity_count: entityTotal,
                highlight_count: highlightTotal,
                highlight_count_returned: highlights.length,
                entities_preview: entities.slice(0, MAX_ENTITIES_PREVIEW),
            };
            var text = JSON.stringify(preview, null, 2);
            if (entityTotal > MAX_ENTITIES_PREVIEW) {
                text += "\\n... entities preview truncated (" + Math.min(MAX_ENTITIES_PREVIEW, entities.length) + "/" + entityTotal + ")";
            }
            document.getElementById('extract-result').innerText = text;
        }
        function setScanPreview(data) {
            if (!data || data.error) {
                document.getElementById('scan-result').innerText = JSON.stringify(data, null, 2);
                return;
            }
            var matches = data.matches || [];
            var previewMatches = matches.slice(0, MAX_MATCHES_PREVIEW).map(function(m) {
                return {
                    dxf_x: m.dxf_x,
                    dxf_y: m.dxf_y,
                    render_pct_x: m.render_pct_x,
                    render_pct_y: m.render_pct_y,
                    match_score: m.match_score,
                    highlight_count_total: m.highlight_count_total ?? (m.highlights ? m.highlights.length : 0),
                    highlight_count_returned: m.highlight_count_returned ?? (m.highlights ? m.highlights.length : 0),
                };
            });
            var preview = {
                match_count: data.match_count || matches.length,
                scan_stats: data.scan_stats || null,
                matches_preview: previewMatches,
            };
            var text = JSON.stringify(preview, null, 2);
            if (matches.length > MAX_MATCHES_PREVIEW) {
                text += "\\n... matches preview truncated (" + MAX_MATCHES_PREVIEW + "/" + matches.length + ")";
            }
            document.getElementById('scan-result').innerText = text;
        }
        function downloadJson(filename, payload) {
            if (!payload) return;
            var text = JSON.stringify(payload, null, 2);
            var blob = new Blob([text], { type: 'application/json' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        async function downloadTemplateFingerprint() {
            if (!currentTemplate) return;
            if (currentTemplate.template_id && currentTemplate.cache_id) {
                var res = await fetch('/template', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        cache_id: currentTemplate.cache_id,
                        template_id: currentTemplate.template_id,
                    })
                });
                var data = await res.json();
                if (data.error) {
                    document.getElementById('extract-result').innerText += "\\n" + data.error;
                    return;
                }
                downloadJson('template_fingerprint.json', data);
                return;
            }
            downloadJson('template_fingerprint.json', currentTemplate);
        }
        function downloadMatchResults() {
            if (!currentMatchResult) return;
            downloadJson('match_results.json', currentMatchResult);
        }
        function collectDropEntityTypes() {
            var selected = Array.from(document.querySelectorAll('.drop-type-opt:checked'))
                .map(function(el) { return (el.value || '').trim().toUpperCase(); })
                .filter(Boolean);
            var customRaw = (document.getElementById('drop-type-custom').value || '');
            var custom = customRaw.split(/[\\s,;]+/)
                .map(function(s) { return s.trim().toUpperCase(); })
                .filter(Boolean);
            var merged = selected.concat(custom);
            return Array.from(new Set(merged));
        }

        // --- Upload ---
        async function uploadDXF() {
            var fi = document.getElementById('dxf-file');
            if (!fi.files[0]) return alert("Please select a file");
            var fastBuild = document.getElementById('fast-cache-build').checked;
            var dropNoisy = document.getElementById('drop-noisy-types').checked;
            var dropMostCommonCircle = document.getElementById('drop-most-common-circle').checked;
            var dropTypes = dropNoisy ? collectDropEntityTypes() : [];
            var fd = new FormData(); fd.append("file", fi.files[0]);
            fd.append("fast_build", fastBuild ? "true" : "false");
            fd.append("drop_noisy_types", dropNoisy ? "true" : "false");
            fd.append("drop_entity_types", JSON.stringify(dropTypes));
            fd.append("drop_most_common_circle", dropMostCommonCircle ? "true" : "false");
            uploadResult.innerText = "Uploading and converting to SVG...";
            clearAll();
            var res = await fetch('/upload', { method: 'POST', body: fd });
            var data = await res.json();
            if (data.svg && data.cache_id) {
                currentCacheId = data.cache_id;
                svgDisplay.innerHTML = data.svg;
                requestAnimationFrame(function() {
                    syncCanvasSize();
                    var mode = data.build_mode || "accurate";
                    var typeCountsBefore = data.entity_type_counts_before_drop || {};
                    var typeCountsAfter = data.entity_type_counts_after_drop || {};
                    var lines = [
                        "Upload successful (" + mode + " build).",
                        "Entity count: " + (data.entity_count ?? 0),
                        "Entity types (before drop): " + JSON.stringify(typeCountsBefore),
                        "Entity types (after drop): " + JSON.stringify(typeCountsAfter),
                        "Drop noisy types: " + (data.drop_noisy_types_enabled ? "ON" : "OFF"),
                        "Drop target types: " + (data.drop_selected_types ? data.drop_selected_types.join(",") : "[]"),
                        "Dropped entity count: " + (data.drop_removed_count ?? 0),
                        "Dropped type details: " + (data.drop_removed_by_type ? JSON.stringify(data.drop_removed_by_type) : "{}"),
                        "Plugin drop-most-common-circle: " + (data.drop_most_common_circle_enabled ? "ON" : "OFF"),
                        "Plugin removed/radius/group: " + (data.drop_most_common_circle_removed ?? 0) + "/" + (data.drop_most_common_circle_radius ?? "null") + "/" + (data.drop_most_common_circle_bucket_count ?? 0),
                        "INSERT scanned/fixed/removed: " + (data.insert_scanned ?? 0) + "/" + (data.insert_fixed ?? 0) + "/" + (data.insert_removed ?? 0),
                        "Coordinate basis: " + (data.coord_basis || "dxf_extents"),
                        "SVG render time: " + (data.svg_render_time_ms ?? 0) + " ms",
                        "Cache build time: " + (data.cache_build_time_ms ?? 0) + " ms",
                    ];
                    uploadResult.innerText = lines.join("\\n");
                    document.getElementById('extract-result').innerText = "Upload successful. Click to add vertices and select a feature.";
                });
            } else {
                currentCacheId = null;
                uploadResult.innerText = JSON.stringify(data, null, 2);
            }
        }
        window.addEventListener('resize', function() {
            if (!contentW || !contentH) return;
            syncCanvasSize();
        });

        // --- Pan (right / middle mouse) ---
        container.addEventListener('mousedown', function(e) {
            if (e.button === 2 || e.button === 1) {
                isPanning = true;
                panLastX = e.clientX; panLastY = e.clientY;
                container.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });
        window.addEventListener('mousemove', function(e) {
            if (isPanning) {
                panX += e.clientX - panLastX; panY += e.clientY - panLastY;
                panLastX = e.clientX; panLastY = e.clientY;
                applyTransform(); return;
            }
            if (polyPts.length > 0 && !polyClosed) drawPoly(e);
        });
        window.addEventListener('mouseup', function() {
            if (isPanning) { isPanning = false; container.style.cursor = 'crosshair'; }
        });
        container.addEventListener('contextmenu', function(e) { e.preventDefault(); });

        // --- Zoom (wheel) ---
        container.addEventListener('wheel', function(e) {
            e.preventDefault();
            var r = container.getBoundingClientRect();
            var mx = e.clientX - r.left, my = e.clientY - r.top;
            var f = e.deltaY < 0 ? 1.1 : 1 / 1.1;
            var ns = Math.max(0.1, Math.min(scale * f, 50));
            panX = mx - (mx - panX) * (ns / scale);
            panY = my - (my - panY) * (ns / scale);
            scale = ns;
            applyTransform();
        }, { passive: false });

        // --- Polygon selection ---
        container.addEventListener('click', function(e) {
            if (e.button !== 0 || isPanning || !contentW) return;
            if (e.detail >= 2) return;
            if (polyClosed) return;
            var pt = screenToContent(e.clientX, e.clientY);
            if (polyPts.length >= 3) {
                var f = polyPts[0];
                var dx = (pt.x - f.x) * scale, dy = (pt.y - f.y) * scale;
                if (Math.sqrt(dx * dx + dy * dy) < 12) { closePoly(); return; }
            }
            polyPts.push(pt);
            drawPoly(e);
        });
        container.addEventListener('dblclick', function(e) {
            e.preventDefault();
            if (!polyClosed && polyPts.length >= 3) closePoly();
        });
        document.addEventListener('keydown', function(e) { if (e.key === 'Escape') clearPoly(); });

        function drawPoly(e) {
            var NS = "http://www.w3.org/2000/svg";
            polyG.innerHTML = '';
            if (!polyPts.length) return;
            var sw = 2 / scale;
            var i, ln, c;

            if (polyClosed) {
                var pg = document.createElementNS(NS, "polygon");
                pg.setAttribute("points", polyPts.map(function(p){ return p.x+','+p.y; }).join(' '));
                pg.setAttribute("fill", "rgba(0,123,255,0.15)");
                pg.setAttribute("stroke", "#007bff"); pg.setAttribute("stroke-width", sw);
                polyG.appendChild(pg);
            } else {
                for (i = 1; i < polyPts.length; i++) {
                    ln = document.createElementNS(NS, "line");
                    ln.setAttribute("x1", polyPts[i-1].x); ln.setAttribute("y1", polyPts[i-1].y);
                    ln.setAttribute("x2", polyPts[i].x);   ln.setAttribute("y2", polyPts[i].y);
                    ln.setAttribute("stroke", "#007bff"); ln.setAttribute("stroke-width", sw);
                    polyG.appendChild(ln);
                }
                if (e) {
                    var cur = screenToContent(e.clientX, e.clientY);
                    var last = polyPts[polyPts.length - 1];
                    ln = document.createElementNS(NS, "line");
                    ln.setAttribute("x1", last.x); ln.setAttribute("y1", last.y);
                    ln.setAttribute("x2", cur.x);  ln.setAttribute("y2", cur.y);
                    ln.setAttribute("stroke", "#007bff"); ln.setAttribute("stroke-width", sw);
                    ln.setAttribute("stroke-dasharray", 4 / scale);
                    polyG.appendChild(ln);
                }
            }
            var vr = 4 / scale;
            for (i = 0; i < polyPts.length; i++) {
                c = document.createElementNS(NS, "circle");
                c.setAttribute("cx", polyPts[i].x); c.setAttribute("cy", polyPts[i].y);
                c.setAttribute("r", (i === 0 && polyPts.length >= 3) ? vr * 1.5 : vr);
                c.setAttribute("fill", (i === 0 && polyPts.length >= 3) ? "#28a745" : "#007bff");
                c.setAttribute("stroke", "white"); c.setAttribute("stroke-width", 1 / scale);
                polyG.appendChild(c);
            }
        }

        async function closePoly() {
            if (!currentCacheId) {
                document.getElementById('extract-result').innerText = "Session expired. Please upload DXF again.";
                return;
            }
            polyClosed = true;
            drawPoly(null);
            currentMatchResult = null;
            btnDownloadMatches.disabled = true;
            var pct = polyPts.map(function(p){ return contentToSvgPct(p); });
            document.getElementById('extract-result').innerText = "Extracting features...";
            var res = await fetch('/extract', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cache_id: currentCacheId, polygon_pct: pct })
            });
            var result = await res.json();
            setExtractPreview(result);
            if (result.error) {
                tmplG.innerHTML = '';
                document.getElementById('btn-scan').disabled = true;
                btnDownloadTemplate.disabled = true;
            } else if ((result.entity_count || 0) > 0 && result.template_id) {
                currentTemplate = {
                    cache_id: result.cache_id || currentCacheId,
                    template_id: result.template_id,
                    group_center: result.group_center,
                    entity_count: result.entity_count || 0,
                    entities_preview: result.entities_preview || [],
                };
                renderTemplateHighlights(result.highlights || []);
                document.getElementById('btn-scan').disabled = false;
                btnDownloadTemplate.disabled = false;
            } else {
                tmplG.innerHTML = '';
                document.getElementById('extract-result').innerText += "\\nNo valid entities found in selection.";
                document.getElementById('btn-scan').disabled = true;
                btnDownloadTemplate.disabled = true;
            }
        }

        function renderTemplateHighlights(highlights) {
            var NS = "http://www.w3.org/2000/svg";
            var sw = 2 / scale;
            tmplG.innerHTML = "";
            var drawList = (highlights || []).slice(0, MAX_TEMPLATE_HIGHLIGHTS_TO_DRAW);
            drawList.forEach(function(h) {
                if (h.kind === "circle") {
                    var c = document.createElementNS(NS, "circle");
                    var cc = svgPctToContent(h.cx_pct, h.cy_pct);
                    c.setAttribute("cx", cc.x);
                    c.setAttribute("cy", cc.y);
                    c.setAttribute("r", Math.max(h.r_pct * svgViewport.w, 2 / scale));
                    c.setAttribute("fill", "rgba(0, 200, 83, 0.12)");
                    c.setAttribute("stroke", "#00c853");
                    c.setAttribute("stroke-width", sw);
                    tmplG.appendChild(c);
                } else if (h.kind === "polyline") {
                    var pl = document.createElementNS(NS, "polyline");
                    var pts = samplePoints(h.points_pct, MAX_POLYLINE_POINTS_DRAW).map(function(pt) {
                        var p = svgPctToContent(pt[0], pt[1]);
                        return p.x + "," + p.y;
                    });
                    pl.setAttribute("points", pts.join(" "));
                    pl.setAttribute("fill", "none");
                    pl.setAttribute("stroke", "#00c853");
                    pl.setAttribute("stroke-width", sw);
                    tmplG.appendChild(pl);
                }
            });
            if (highlights && highlights.length > MAX_TEMPLATE_HIGHLIGHTS_TO_DRAW) {
                document.getElementById('extract-result').innerText +=
                    "\\nHighlight draw capped: " + MAX_TEMPLATE_HIGHLIGHTS_TO_DRAW + "/" + highlights.length;
            }
        }

        function clearPoly() {
            polyPts = []; polyClosed = false; polyG.innerHTML = '';
            tmplG.innerHTML = '';
            currentTemplate = null;
            currentMatchResult = null;
            document.getElementById('btn-scan').disabled = true;
            btnDownloadTemplate.disabled = true;
            btnDownloadMatches.disabled = true;
            document.getElementById('extract-result').innerText = "Cleared. Click to start a new selection.";
        }
        function resetView() { scale = 1; panX = 0; panY = 0; applyTransform(); }
        function clearAll() {
            clearPoly(); hlG.innerHTML = '';
            currentCacheId = null;
            resetView();
            document.getElementById('scan-status').innerText = '';
            document.getElementById('scan-result').innerText = 'Waiting for scan...';
        }

        // --- Scan ---
        async function scanTemplate() {
            if (!currentTemplate || !currentCacheId || !currentTemplate.template_id) return;
            var endpoint = document.querySelector('input[name="scan-mode"]:checked').value;
            document.getElementById('scan-status').innerText = "Scanning...";
            document.getElementById('scan-result').innerText = "";
            hlG.innerHTML = "";
            var t0 = performance.now();
            var payload = {
                cache_id: currentCacheId,
                template_id: currentTemplate.template_id,
            };
            var res = await fetch(endpoint, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            var data = await res.json();
            var elapsed = ((performance.now() - t0) / 1000).toFixed(2);
            if (data.error) {
                document.getElementById('scan-status').innerText = data.error;
                document.getElementById('scan-result').innerText = JSON.stringify(data, null, 2);
                currentMatchResult = null;
                btnDownloadMatches.disabled = true;
                return;
            }
            document.getElementById('scan-status').innerText = 'Scan complete! Found ' + data.match_count + ' matches (' + elapsed + 's)';
            var noHighlightMatches = (data.matches || []).filter(function(m) {
                return (m.highlight_count_total || 0) > 0 && (m.highlight_count_returned || 0) === 0;
            }).length;
            if (noHighlightMatches > 0) {
                document.getElementById('scan-status').innerText +=
                    ' | Server highlight payload capped for ' + noHighlightMatches + ' matches';
            }
            if (data.scan_stats && data.scan_stats.score_max !== undefined) {
                document.getElementById('scan-status').innerText +=
                    ' | score_max=' + data.scan_stats.score_max;
            }
            setScanPreview(data);
            renderHighlights(data.matches);
            currentMatchResult = data;
            btnDownloadMatches.disabled = false;
        }

        function renderHighlights(matches) {
            var NS = "http://www.w3.org/2000/svg";
            var sw = 2 / scale;
            hlG.innerHTML = "";
            var drawMatches = (matches || []).slice(0, MAX_MATCHES_TO_DRAW);
            var drawnHighlights = 0;
            var droppedPerMatch = 0;
            var droppedGlobal = 0;
            drawMatches.forEach(function(match) {
                if (!match.highlights) return;
                if (drawnHighlights >= MAX_TOTAL_HIGHLIGHTS_TO_DRAW) {
                    droppedGlobal += match.highlights.length;
                    return;
                }
                var perMatch = match.highlights.slice(0, MAX_HIGHLIGHTS_PER_MATCH_TO_DRAW);
                droppedPerMatch += Math.max(0, match.highlights.length - perMatch.length);
                var remaining = MAX_TOTAL_HIGHLIGHTS_TO_DRAW - drawnHighlights;
                var drawHs = perMatch.slice(0, remaining);
                droppedGlobal += Math.max(0, perMatch.length - drawHs.length);
                drawHs.forEach(function(h) {
                    if (h.kind === "circle") {
                        var c = document.createElementNS(NS, "circle");
                        var cc = svgPctToContent(h.cx_pct, h.cy_pct);
                        c.setAttribute("cx", cc.x);
                        c.setAttribute("cy", cc.y);
                        c.setAttribute("r", Math.max(h.r_pct * svgViewport.w, 2 / scale));
                        c.setAttribute("fill", "none");
                        c.setAttribute("stroke", "red"); c.setAttribute("stroke-width", sw);
                        hlG.appendChild(c);
                    } else if (h.kind === "polyline") {
                        var pl = document.createElementNS(NS, "polyline");
                        var pts = samplePoints(h.points_pct, MAX_POLYLINE_POINTS_DRAW).map(function(pt) {
                            var p = svgPctToContent(pt[0], pt[1]);
                            return p.x + "," + p.y;
                        });
                        pl.setAttribute("points", pts.join(" "));
                        pl.setAttribute("fill", "none");
                        pl.setAttribute("stroke", "red");
                        pl.setAttribute("stroke-width", sw);
                        hlG.appendChild(pl);
                    }
                });
                drawnHighlights += drawHs.length;
            });
            if (matches && matches.length > MAX_MATCHES_TO_DRAW) {
                document.getElementById('scan-status').innerText +=
                    " | Highlight draw capped: " + MAX_MATCHES_TO_DRAW + "/" + matches.length;
            }
            if (droppedPerMatch > 0 || droppedGlobal > 0) {
                document.getElementById('scan-status').innerText +=
                    " | Draw detail capped (per-match: " + droppedPerMatch + ", global: " + droppedGlobal + ")";
            }
        }
    </script>
</body>
</html>
"""

# --- FastAPI routes ---


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content


@app.post("/upload")
async def upload_dxf(
    file: UploadFile = File(...),
    fast_build: bool = Form(False),
    drop_noisy_types: bool = Form(True),
    drop_entity_types: str = Form(""),
    drop_most_common_circle: bool = Form(False),
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

    type_counts_before_drop = _count_entity_types(doc.modelspace())
    drop_removed_count = 0
    drop_removed_by_type = {}
    drop_selected_types = _parse_entity_types_form(drop_entity_types)
    if not drop_selected_types:
        drop_selected_types = sorted(DEFAULT_DROP_ENTITY_TYPES)
    if drop_noisy_types:
        drop_removed_count, drop_removed_by_type = _drop_entities_by_type(
            doc, drop_selected_types
        )

    circle_plugin_result = {
        "enabled": bool(drop_most_common_circle),
        "applied": False,
        "removed_count": 0,
        "radius": None,
        "bucket_count": 0,
        "unique_radius_count": 0,
    }
    if drop_most_common_circle:
        circle_plugin_result = _run_upload_doc_plugin(
            "drop_most_common_circle_size",
            doc,
            {
                "decimals": CIRCLE_MODE_RADIUS_DECIMALS,
                "min_group_count": CIRCLE_MODE_MIN_GROUP_COUNT,
            },
        )

    msp = doc.modelspace()
    dxf_entity_count = len(msp)
    type_counts_after_drop = _count_entity_types(msp)
    insert_scanned, insert_fixed = _sanitize_doc_inserts(doc)
    backend = SVGBackend()
    render_config = Configuration.defaults().with_changes(
        lineweight_scaling=SVG_LINEWEIGHT_SCALING
    )

    t0 = time.perf_counter()
    render_retry_removed_inserts = 0
    try:
        Frontend(RenderContext(doc), backend, config=render_config).draw_layout(msp)
    except Exception:
        # Some DWG->DXF files contain degenerate INSERT transforms.
        # Drop only INSERT entities that cannot be virtually expanded, then retry once.
        render_retry_removed_inserts = _drop_unrenderable_inserts(doc)
        msp = doc.modelspace()
        backend = SVGBackend()
        try:
            Frontend(RenderContext(doc), backend, config=render_config).draw_layout(msp)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Failed to render DXF: {str(exc)}",
                    "drop_noisy_types_enabled": bool(drop_noisy_types),
                    "drop_selected_types": drop_selected_types if drop_noisy_types else [],
                    "drop_removed_count": int(drop_removed_count),
                    "drop_removed_by_type": drop_removed_by_type,
                    "entity_type_counts_before_drop": type_counts_before_drop,
                    "entity_type_counts_after_drop": type_counts_after_drop,
                    "drop_most_common_circle_enabled": bool(
                        circle_plugin_result.get("enabled", False)
                    ),
                    "drop_most_common_circle_applied": bool(
                        circle_plugin_result.get("applied", False)
                    ),
                    "drop_most_common_circle_removed": int(
                        circle_plugin_result.get("removed_count", 0)
                    ),
                    "drop_most_common_circle_radius": circle_plugin_result.get("radius"),
                    "drop_most_common_circle_bucket_count": int(
                        circle_plugin_result.get("bucket_count", 0)
                    ),
                    "drop_most_common_circle_unique_radius_count": int(
                        circle_plugin_result.get("unique_radius_count", 0)
                    ),
                    "drop_most_common_circle_error": circle_plugin_result.get("error"),
                    "insert_scanned": insert_scanned,
                    "insert_fixed": insert_fixed,
                    "insert_removed": render_retry_removed_inserts,
                },
            )
    render_bounds = None
    render_mapping = None
    try:
        player_bbox = backend.player().bbox()
        if player_bbox.has_data:
            render_bounds = _bbox2d_to_bounds(player_bbox)
    except Exception:
        render_bounds = None
    page = layout.Page(0, 0, layout.Units.mm)
    svg = backend.get_string(page)
    try:
        inv = backend.transformation_matrix.copy()
        inv.inverse()
        rows = list(inv.rows())
        inv_flat = [float(v) for row in rows for v in row]
        root = ET.fromstring(svg)
        vb = (root.get("viewBox") or "").strip().split()
        if len(vb) == 4:
            viewbox_w = float(vb[2])
            viewbox_h = float(vb[3])
            if viewbox_w > 0 and viewbox_h > 0:
                render_mapping = {
                    "inv_matrix": inv_flat,
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
        bounds_override=render_bounds,
        render_mapping=render_mapping,
    )
    cache_build_time_ms = int((time.perf_counter() - t1) * 1000)
    cache_id = _store_cache(cache_payload)
    return {
        "status": "success",
        "cache_id": cache_id,
        "build_mode": "fast" if fast_build else "accurate",
        "drop_noisy_types_enabled": bool(drop_noisy_types),
        "drop_selected_types": drop_selected_types if drop_noisy_types else [],
        "drop_removed_count": int(drop_removed_count),
        "drop_removed_by_type": drop_removed_by_type,
        "entity_type_counts_before_drop": type_counts_before_drop,
        "entity_type_counts_after_drop": type_counts_after_drop,
        "drop_most_common_circle_enabled": bool(
            circle_plugin_result.get("enabled", False)
        ),
        "drop_most_common_circle_applied": bool(
            circle_plugin_result.get("applied", False)
        ),
        "drop_most_common_circle_removed": int(
            circle_plugin_result.get("removed_count", 0)
        ),
        "drop_most_common_circle_radius": circle_plugin_result.get("radius"),
        "drop_most_common_circle_bucket_count": int(
            circle_plugin_result.get("bucket_count", 0)
        ),
        "drop_most_common_circle_unique_radius_count": int(
            circle_plugin_result.get("unique_radius_count", 0)
        ),
        "drop_most_common_circle_error": circle_plugin_result.get("error"),
        "dxf_entity_count": dxf_entity_count,
        "entity_count": len(cache_payload["fingerprints"]),
        "svg_render_time_ms": svg_render_time_ms,
        "cache_build_time_ms": cache_build_time_ms,
        "insert_scanned": insert_scanned,
        "insert_fixed": insert_fixed,
        "insert_removed": render_retry_removed_inserts,
        "coord_basis": (
            "render_matrix"
            if render_mapping
            else ("render_bbox" if render_bounds else "dxf_extents")
        ),
        "svg": svg,
    }


@app.post("/extract")
async def extract_template(data: dict = Body(...)):
    cache_id = data.get("cache_id")
    cache_payload = _get_cache(cache_id)
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or expired cache_id. Please upload a file again."},
        )

    polygon_pct = data.get("polygon_pct", [])
    if len(polygon_pct) < 3:
        return {"entities": [], "highlights": []}

    bounds = cache_payload["bounds"]

    # Convert percentage vertices to DXF coordinates.
    # Preferred: exact inverse render matrix (robust for transformed/converted DXF).
    dxf_vertices = []
    render_mapping = cache_payload.get("render_mapping") or {}
    inv_flat = render_mapping.get("inv_matrix")
    viewbox_w = render_mapping.get("viewbox_w")
    viewbox_h = render_mapping.get("viewbox_h")
    if inv_flat and viewbox_w and viewbox_h:
        try:
            inv = Matrix44(inv_flat)
            dxf_vertices = []
            for pt in polygon_pct:
                x_vb = float(pt[0]) * float(viewbox_w)
                y_vb = float(pt[1]) * float(viewbox_h)
                p = inv.transform((x_vb, y_vb, 0.0))
                dxf_vertices.append((float(p[0]), float(p[1])))
        except Exception:
            dxf_vertices = []

    if not dxf_vertices:
        # Fallback: linear bounds mapping.
        dxf_vertices = [
            (
                bounds["min_x"] + pt[0] * bounds["width"],
                bounds["max_y"] - pt[1] * bounds["height"],
            )
            for pt in polygon_pct
        ]
    roi_polygon = ShapelyPolygon(dxf_vertices)

    # Filter directly from cache (no linemerge recomputation)
    entities_found = [
        f
        for f in cache_payload["fingerprints"]
        if roi_polygon.covers(Point(f["x"], f["y"]))
    ]

    if not entities_found:
        return {"entities": [], "highlights": []}

    # Compute group centroid
    group_cx = sum(e["x"] for e in entities_found) / len(entities_found)
    group_cy = sum(e["y"] for e in entities_found) / len(entities_found)

    # Remove geometry field to keep template JSON compact
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
    all_highlights = [geometry_to_render_pct(e["geometry"], bounds) for e in entities_found]
    highlights = [
        _compact_render_highlight(h)
        for h in all_highlights[:MAX_EXTRACT_HIGHLIGHTS_RETURN]
    ]

    return {
        "cache_id": cache_id,
        "template_id": template_id,
        "group_center": template_payload["group_center"],
        "entity_count": len(entities_clean),
        "entities_preview": entities_clean[:MAX_EXTRACT_ENTITIES_PREVIEW],
        "highlights": highlights,
        "highlight_count_total": len(all_highlights),
        "highlight_count_returned": len(highlights),
        "highlight_truncated": len(all_highlights) > len(highlights),
    }


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
    hl = [_compact_render_highlight(geometry_to_render_pct(f["geometry"], bounds)) for f in selected]

    return {
        "dxf_x": round(mcx, 3),
        "dxf_y": round(mcy, 3),
        "render_pct_x": round(rpx, 4),
        "render_pct_y": round(rpy, 4),
        "match_score": (round(float(match_score), 6) if match_score is not None else None),
        "highlights": hl,
        "highlight_count_total": len(mf),
        "highlight_count_returned": len(hl),
        "highlight_truncated": len(hl) < len(mf),
    }


def _trim_match_highlights(matches, max_with_highlights=MAX_SCAN_MATCHES_WITH_HIGHLIGHTS):
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
    cache_payload = _get_cache(template.get("cache_id"))
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or expired cache_id. Please upload a file again."},
        )

    bounds = cache_payload["bounds"]
    all_fp = cache_payload["fingerprints"]
    tree = cache_payload["tree"]
    type_index = cache_payload["type_index"]
    sizes = cache_payload["sizes"]

    template_id = template.get("template_id")
    if template_id:
        stored_template = _load_extracted_template(template.get("cache_id"), template_id)
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

    score_max = _resolve_score_max(template)
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
        anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)
        stats["candidate_anchor_base"] = sum(
            1
            for f in all_fp
            if f["type"] == anchor_tmpl["type"]
            and abs(f["size"] - anchor_tmpl["size"]) < anchor_size_tol
        )
        t_prefilter = time.perf_counter()
        matched_indices = _single_entity_matches(all_fp, anchor_tmpl)
        stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)
        stats["candidate_anchor_after_plugin"] = len(matched_indices)
        t_matching = time.perf_counter()
        matches_found = []
        for i in matched_indices:
            f = all_fp[i]
            s = _pair_match_score(anchor_tmpl, f, size_tol=anchor_size_tol)
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
                    match_score=s,
                )
            )
        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
        stats["raw_match_count"] = len(matches_found)
        unique = _dedupe_and_sort_matches(matches_found)
        stats["unique_match_count"] = len(unique)
        return {
            "match_count": len(unique),
            "matches": _trim_match_highlights(unique),
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    # Pick the rarest entity as anchor
    t_prefilter = time.perf_counter()
    anchor_idx = _pick_anchor(entities, all_fp, type_index, sizes)
    anchor_tmpl = entities[anchor_idx]
    others = [entities[i] for i in range(len(entities)) if i != anchor_idx]

    anchor_pt = (anchor_tmpl["x"], anchor_tmpl["y"])
    anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)

    targets = []
    for o in others:
        d = calculate_distance(anchor_pt, (o["x"], o["y"]))
        targets.append(
            {
                "entity": o,
                "type": o["type"],
                "size": o["size"],
                "size_tol": max(SIZE_TOL_MIN, o["size"] * SIZE_TOL_RATIO),
                "dist": d,
                "dist_tol": max(DIST_TOL_MIN, d * DIST_TOL_RATIO),
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
        (i, f) for i, f in potential_base if _single_entity_plugin_pass(anchor_tmpl, f)
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
                if not _entity_basic_match(t["entity"], f, size_tol=t["size_tol"]):
                    continue
                d = calculate_distance(cp, (f["x"], f["y"]))
                dist_err = abs(d - t["dist"])
                if dist_err < t["dist_tol"]:
                    edge_score = _pair_match_score(
                        t["entity"],
                        f,
                        size_tol=t["size_tol"],
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
                anchor_tmpl, ac, size_tol=anchor_size_tol
            )
            total_score = (
                MATCH_SCORE_ANCHOR_WEIGHT * anchor_score
                + (1.0 - MATCH_SCORE_ANCHOR_WEIGHT) * edge_mean_score
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
        "matches": _trim_match_highlights(unique),
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
    cache_payload = _get_cache(template.get("cache_id"))
    if cache_payload is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or expired cache_id. Please upload a file again."},
        )

    bounds = cache_payload["bounds"]
    all_fp = cache_payload["fingerprints"]
    coords = cache_payload["coords"]
    sizes = cache_payload["sizes"]
    tree = cache_payload["tree"]
    type_index = cache_payload["type_index"]

    template_id = template.get("template_id")
    if template_id:
        stored_template = _load_extracted_template(template.get("cache_id"), template_id)
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

    score_max = _resolve_score_max(template)
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
        anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)
        stats["candidate_anchor_base"] = sum(
            1
            for f in all_fp
            if f["type"] == anchor_tmpl["type"]
            and abs(f["size"] - anchor_tmpl["size"]) < anchor_size_tol
        )
        t_prefilter = time.perf_counter()
        matched_indices = _single_entity_matches(all_fp, anchor_tmpl)
        stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)
        stats["candidate_anchor_after_plugin"] = len(matched_indices)

        t_matching = time.perf_counter()
        matches_found = []
        for ai in matched_indices:
            f = all_fp[int(ai)]
            s = _pair_match_score(anchor_tmpl, f, size_tol=anchor_size_tol)
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
                    match_score=s,
                )
            )
        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
        stats["raw_match_count"] = len(matches_found)
        unique = _dedupe_and_sort_matches(matches_found)
        stats["unique_match_count"] = len(unique)
        return {
            "match_count": len(unique),
            "matches": _trim_match_highlights(unique),
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    # Pick the rarest entity as anchor
    t_prefilter = time.perf_counter()
    anchor_idx = _pick_anchor(entities, all_fp, type_index, sizes)
    anchor_tmpl = entities[anchor_idx]
    others = [entities[i] for i in range(len(entities)) if i != anchor_idx]

    anchor_pt = np.array([anchor_tmpl["x"], anchor_tmpl["y"]])
    anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)

    targets = []
    for o in others:
        d = float(np.linalg.norm(anchor_pt - [o["x"], o["y"]]))
        targets.append(
            {
                "entity": o,
                "type": o["type"],
                "size": o["size"],
                "size_tol": max(SIZE_TOL_MIN, o["size"] * SIZE_TOL_RATIO),
                "dist": d,
                "dist_tol": max(DIST_TOL_MIN, d * DIST_TOL_RATIO),
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
            if _single_entity_plugin_pass(anchor_tmpl, all_fp[int(i)])
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
                    if _single_entity_plugin_pass(t["entity"], all_fp[int(ci)])
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
                anchor_tmpl, all_fp[int(ai)], size_tol=anchor_size_tol
            )
            total_score = (
                MATCH_SCORE_ANCHOR_WEIGHT * anchor_score
                + (1.0 - MATCH_SCORE_ANCHOR_WEIGHT) * edge_mean_score
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
        "matches": _trim_match_highlights(unique),
        "scan_stats": _finalize_scan_stats(stats, total_start),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
