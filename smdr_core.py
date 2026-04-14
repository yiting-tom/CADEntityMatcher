import json
import math
import time
import uuid
from collections import Counter
from typing import Callable

import ezdxf
import numpy as np
from ezdxf.bbox import extents
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon
from shapely.ops import linemerge

SNAP_DECIMALS = 6
SIZE_TOL_RATIO = 0.05
SIZE_TOL_MIN = 0.05
DIST_TOL_RATIO = 0.05
DIST_TOL_MIN = 0.1
CACHE_TTL_SECONDS = 60 * 60
SINGLE_POINT_COUNT_TOL = 2
SINGLE_ASPECT_RATIO_TOL = 0.20
SINGLE_DIAG_TOL = 0.10
SINGLE_EDGE_HIST_BINS = 8
SINGLE_TURN_HIST_BINS = 12
SINGLE_EDGE_HIST_L1_TOL = 0.55
SINGLE_TURN_HIST_L1_TOL = 0.60
DEFAULT_FLATTEN_TOL = 0.01
FAST_FLATTEN_TOL = 0.08
MAX_EXTRACT_HIGHLIGHTS_RETURN = 250
MAX_HIGHLIGHT_POINTS_RETURN = 400
MAX_EXTRACT_ENTITIES_PREVIEW = 80
MAX_TEMPLATES_PER_SESSION = 16
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
    return round(val, SNAP_DECIMALS)


def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _compact_geometry(geometry):
    if geometry.get("kind") != "polyline":
        return geometry
    compact = dict(geometry)
    compact["points"] = _sample_points(geometry.get("points", []), MAX_HIGHLIGHT_POINTS_RETURN)
    return compact


def dxf_to_scene_point(x, y):
    return float(x), -float(y)


def scene_to_dxf_point(x, y):
    return float(x), -float(y)


def dxf_points_to_scene(points):
    return [dxf_to_scene_point(x, y) for x, y in points]


def scene_points_to_dxf(points):
    return [scene_to_dxf_point(x, y) for x, y in points]


def geometry_to_scene(geometry):
    if geometry["kind"] == "circle":
        sx, sy = dxf_to_scene_point(geometry["cx"], geometry["cy"])
        return {"kind": "circle", "cx": sx, "cy": sy, "r": float(geometry["r"])}
    return {
        "kind": "polyline",
        "points": [list(dxf_to_scene_point(px, py)) for px, py in geometry["points"]],
    }


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
    ext = extents(msp)
    return _make_bounds(ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y)


def bounds_to_scene_rect(bounds):
    return (
        bounds["min_x"],
        -bounds["max_y"],
        bounds["width"],
        bounds["height"],
    )


def _iter_flat_entities(entities, *, depth=0, max_depth=MAX_INSERT_EXPLODE_DEPTH):
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
    yield doc.modelspace()
    for blk in doc.blocks:
        yield blk


def _sorted_type_counts(counter):
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _count_entity_types(entities):
    counts = Counter()
    for entity in entities:
        try:
            counts[entity.dxftype().upper()] += 1
        except Exception:
            continue
    return _sorted_type_counts(counts)


def parse_entity_types(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        values = [str(v) for v in raw]
    else:
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
    for value in values:
        entity_type = str(value).strip().upper()
        if not entity_type or entity_type in seen:
            continue
        seen.add(entity_type)
        out.append(entity_type)
    return out


def _sanitize_doc_inserts(doc):
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


def _drop_entities_by_type(doc, type_names):
    targets = {str(t).upper() for t in type_names}
    removed = 0
    removed_by_type = {}
    for space in _iter_all_entity_spaces(doc):
        for entity in list(space):
            entity_type = entity.dxftype().upper()
            if entity_type not in targets:
                continue
            try:
                space.delete_entity(entity)
                removed += 1
                removed_by_type[entity_type] = removed_by_type.get(entity_type, 0) + 1
            except Exception:
                pass
    return removed, removed_by_type


def _drop_unrenderable_inserts(doc):
    removed = 0
    for space in _iter_all_entity_spaces(doc):
        for entity in list(space):
            if entity.dxftype() != "INSERT":
                continue
            try:
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
    if not points:
        return {
            "point_count": 0,
            "bbox_diag": 0.0,
            "aspect_ratio": 1.0,
            "edge_hist": [0.0] * SINGLE_EDGE_HIST_BINS,
            "turn_hist": [0.0] * SINGLE_TURN_HIST_BINS,
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    long_side = max(width, height)
    short_side = max(min(width, height), 1e-9)
    seg_lengths = []
    turn_angles = []

    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        length = math.hypot(dx, dy)
        if length > BOUNDS_EPS:
            seg_lengths.append(length)

    total_len = sum(seg_lengths)
    if total_len > BOUNDS_EPS:
        seg_norm = [length / total_len for length in seg_lengths]
        edge_hist = (
            np.histogram(seg_norm, bins=SINGLE_EDGE_HIST_BINS, range=(0.0, 1.0))[0]
            .astype(np.float64)
            .tolist()
        )
        edge_sum = sum(edge_hist) or 1.0
        edge_hist = [round(v / edge_sum, 6) for v in edge_hist]
    else:
        edge_hist = [0.0] * SINGLE_EDGE_HIST_BINS

    for i in range(1, len(points) - 1):
        ax = points[i][0] - points[i - 1][0]
        ay = points[i][1] - points[i - 1][1]
        bx = points[i + 1][0] - points[i][0]
        by = points[i + 1][1] - points[i][1]
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la <= BOUNDS_EPS or lb <= BOUNDS_EPS:
            continue
        cosv = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        turn_angles.append(math.acos(cosv))

    if turn_angles:
        turn_hist = (
            np.histogram(turn_angles, bins=SINGLE_TURN_HIST_BINS, range=(0.0, math.pi))[0]
            .astype(np.float64)
            .tolist()
        )
        turn_sum = sum(turn_hist) or 1.0
        turn_hist = [round(v / turn_sum, 6) for v in turn_hist]
    else:
        turn_hist = [0.0] * SINGLE_TURN_HIST_BINS

    return {
        "point_count": int(len(points)),
        "bbox_diag": round(math.hypot(width, height), 6),
        "aspect_ratio": round(long_side / short_side, 6),
        "edge_hist": edge_hist,
        "turn_hist": turn_hist,
    }


def _arc_points(entity):
    center = entity.dxf.center
    radius = entity.dxf.radius
    sa = math.radians(entity.dxf.start_angle)
    ea = math.radians(entity.dxf.end_angle)
    if ea <= sa:
        ea += 2 * math.pi
    n = max(8, int(abs(ea - sa) / (math.pi / 18)))
    return [
        (
            snap(center.x + radius * math.cos(sa + (ea - sa) * i / n)),
            snap(center.y + radius * math.sin(sa + (ea - sa) * i / n)),
        )
        for i in range(n + 1)
    ]


def collect_preview_primitives(
    msp, *, flatten_tol=DEFAULT_FLATTEN_TOL, skip_ellipse_spline=False
):
    primitives = []
    for entity in _iter_flat_entities(msp):
        entity_type = entity.dxftype()

        if entity_type == "CIRCLE":
            primitives.append(
                {
                    "kind": "circle",
                    "cx": float(entity.dxf.center.x),
                    "cy": float(entity.dxf.center.y),
                    "r": float(entity.dxf.radius),
                }
            )
        elif entity_type == "LINE":
            p1, p2 = entity.dxf.start, entity.dxf.end
            primitives.append(
                {
                    "kind": "polyline",
                    "points": [
                        [float(p1.x), float(p1.y)],
                        [float(p2.x), float(p2.y)],
                    ],
                }
            )
        elif entity_type == "LWPOLYLINE":
            points = [(float(p[0]), float(p[1])) for p in entity.get_points()]
            if entity.is_closed and len(points) >= 3:
                points.append(points[0])
            if len(points) >= 2:
                primitives.append({"kind": "polyline", "points": [list(p) for p in points]})
        elif entity_type == "ARC":
            primitives.append({"kind": "polyline", "points": [list(p) for p in _arc_points(entity)]})
        elif entity_type == "POLYLINE":
            try:
                points = [
                    (float(v.dxf.location.x), float(v.dxf.location.y))
                    for v in entity.vertices
                ]
            except Exception:
                points = []
            if entity.is_closed and len(points) >= 3:
                points.append(points[0])
            if len(points) >= 2:
                primitives.append({"kind": "polyline", "points": [list(p) for p in points]})
        elif entity_type in ("ELLIPSE", "SPLINE"):
            if skip_ellipse_spline:
                continue
            try:
                points = [(float(p.x), float(p.y)) for p in entity.flattening(flatten_tol)]
            except Exception:
                points = []
            if len(points) >= 2:
                primitives.append({"kind": "polyline", "points": [list(p) for p in points]})

    return primitives


def extract_template_features(
    msp, roi_box=None, *, flatten_tol=DEFAULT_FLATTEN_TOL, skip_ellipse_spline=False
):
    circles = []
    all_lines = []

    for entity in _iter_flat_entities(msp):
        entity_type = entity.dxftype()

        if entity_type == "CIRCLE":
            cx = float(entity.dxf.center.x)
            cy = float(entity.dxf.center.y)
            circles.append(
                {
                    "type": "CIRCLE",
                    "size": round(float(entity.dxf.radius), 3),
                    "x": round(cx, 3),
                    "y": round(cy, 3),
                    "geometry": {
                        "kind": "circle",
                        "cx": cx,
                        "cy": cy,
                        "r": float(entity.dxf.radius),
                    },
                }
            )

        elif entity_type == "LINE":
            p1, p2 = entity.dxf.start, entity.dxf.end
            all_lines.append(LineString([(snap(p1.x), snap(p1.y)), (snap(p2.x), snap(p2.y))]))

        elif entity_type == "LWPOLYLINE":
            points = [(snap(p[0]), snap(p[1])) for p in entity.get_points()]
            if entity.is_closed and len(points) >= 3:
                points.append(points[0])
            if len(points) >= 2:
                all_lines.append(LineString(points))

        elif entity_type == "ARC":
            all_lines.append(LineString(_arc_points(entity)))

        elif entity_type == "POLYLINE":
            try:
                points = [
                    (snap(v.dxf.location.x), snap(v.dxf.location.y))
                    for v in entity.vertices
                ]
            except Exception:
                points = []
            if entity.is_closed and len(points) >= 3:
                points.append(points[0])
            if len(points) >= 2:
                all_lines.append(LineString(points))

        elif entity_type in ("ELLIPSE", "SPLINE"):
            if skip_ellipse_spline:
                continue
            try:
                points = [(snap(p.x), snap(p.y)) for p in entity.flattening(flatten_tol)]
            except Exception:
                points = []
            if len(points) >= 2:
                all_lines.append(LineString(points))

    features = list(circles)
    if all_lines:
        merged = linemerge(all_lines)
        if merged.geom_type in ("LineString", "LinearRing"):
            merged_geoms = [merged]
        else:
            merged_geoms = list(merged.geoms)

        for geom in merged_geoms:
            centroid = geom.centroid
            features.append(
                {
                    "type": "COMPOSITE_SHAPE",
                    "size": round(float(geom.length), 3),
                    "x": round(float(centroid.x), 3),
                    "y": round(float(centroid.y), 3),
                    "shape_sig": _polyline_shape_signature(list(geom.coords)),
                    "geometry": {
                        "kind": "polyline",
                        "points": [list(c) for c in geom.coords],
                    },
                }
            )

    if roi_box is not None:
        features = [f for f in features if roi_box.covers(Point(f["x"], f["y"]))]

    return features


def _build_session_payload(msp, *, fast_build=False):
    bounds = get_dxf_bounds(msp)
    flatten_tol = FAST_FLATTEN_TOL if fast_build else DEFAULT_FLATTEN_TOL
    skip_ellipse_spline = bool(fast_build)
    fingerprints = extract_template_features(
        msp,
        roi_box=None,
        flatten_tol=flatten_tol,
        skip_ellipse_spline=skip_ellipse_spline,
    )
    preview_primitives = collect_preview_primitives(
        msp,
        flatten_tol=flatten_tol,
        skip_ellipse_spline=skip_ellipse_spline,
    )

    count = len(fingerprints)
    if count > 0:
        coords = np.array([[f["x"], f["y"]] for f in fingerprints], dtype=np.float64)
        sizes = np.array([f["size"] for f in fingerprints], dtype=np.float64)
        tree = cKDTree(coords)
    else:
        coords = np.empty((0, 2), dtype=np.float64)
        sizes = np.empty(0, dtype=np.float64)
        tree = None

    type_index = {}
    for idx, feature in enumerate(fingerprints):
        type_index.setdefault(feature["type"], []).append(idx)
    type_index = {key: np.array(value, dtype=np.intp) for key, value in type_index.items()}

    now = time.time()
    return {
        "session_id": uuid.uuid4().hex,
        "bounds": bounds,
        "fingerprints": fingerprints,
        "preview_primitives": preview_primitives,
        "coords": coords,
        "sizes": sizes,
        "tree": tree,
        "type_index": type_index,
        "templates": {},
        "fast_build": bool(fast_build),
        "flatten_tol": flatten_tol,
        "created_at": now,
        "last_access": now,
    }


def _store_template(session, group_center, entities):
    template_id = uuid.uuid4().hex
    templates = session.setdefault("templates", {})
    templates[template_id] = {
        "group_center": group_center,
        "entities": entities,
        "created_at": time.time(),
    }
    if len(templates) > MAX_TEMPLATES_PER_SESSION:
        oldest = sorted(templates.items(), key=lambda item: item[1].get("created_at", 0))
        for old_template_id, _ in oldest[: len(templates) - MAX_TEMPLATES_PER_SESSION]:
            templates.pop(old_template_id, None)
    session["last_access"] = time.time()
    return template_id


def load_template(session, template_id):
    session["last_access"] = time.time()
    return session.get("templates", {}).get(template_id)


SingleEntityPlugin = Callable[[dict, dict], bool]
SINGLE_ENTITY_MATCH_PLUGINS: list[SingleEntityPlugin] = []

UploadDocPlugin = Callable[[object, dict], dict]
UPLOAD_DOC_PLUGINS: dict[str, UploadDocPlugin] = {}


def register_single_entity_plugin(plugin: SingleEntityPlugin):
    if plugin not in SINGLE_ENTITY_MATCH_PLUGINS:
        SINGLE_ENTITY_MATCH_PLUGINS.append(plugin)


def register_upload_doc_plugin(name: str, plugin: UploadDocPlugin):
    key = str(name).strip()
    if key:
        UPLOAD_DOC_PLUGINS[key] = plugin


def _run_upload_doc_plugin(name: str, doc, options=None):
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


def _hist_l1_distance(a, b):
    if not isinstance(a, list) or not isinstance(b, list):
        return None
    if len(a) == 0 or len(a) != len(b):
        return None
    try:
        return float(sum(abs(float(x) - float(y)) for x, y in zip(a, b)))
    except Exception:
        return None


def _plugin_composite_shape_histogram(template_entity, candidate_feature):
    if template_entity.get("type") != "COMPOSITE_SHAPE":
        return True
    ts = template_entity.get("shape_sig")
    cs = candidate_feature.get("shape_sig")
    if not isinstance(ts, dict) or not isinstance(cs, dict):
        return True

    edge_l1 = _hist_l1_distance(ts.get("edge_hist"), cs.get("edge_hist"))
    if edge_l1 is not None and edge_l1 > SINGLE_EDGE_HIST_L1_TOL:
        return False

    turn_l1 = _hist_l1_distance(ts.get("turn_hist"), cs.get("turn_hist"))
    if turn_l1 is not None and turn_l1 > SINGLE_TURN_HIST_L1_TOL:
        return False
    return True


def _single_entity_plugin_pass(template_entity, candidate_feature):
    for plugin in SINGLE_ENTITY_MATCH_PLUGINS:
        if not plugin(template_entity, candidate_feature):
            return False
    return True


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _shape_signature_error(template_entity, candidate_feature):
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
        parts.append(_clamp01((abs(ca - ta) / ta) / SINGLE_ASPECT_RATIO_TOL))

    td = ts.get("bbox_diag")
    cd = cs.get("bbox_diag")
    if isinstance(td, (int, float)) and isinstance(cd, (int, float)) and td > 0:
        parts.append(_clamp01((abs(cd - td) / td) / SINGLE_DIAG_TOL))

    edge_l1 = _hist_l1_distance(ts.get("edge_hist"), cs.get("edge_hist"))
    if edge_l1 is not None:
        parts.append(_clamp01(edge_l1 / SINGLE_EDGE_HIST_L1_TOL))

    turn_l1 = _hist_l1_distance(ts.get("turn_hist"), cs.get("turn_hist"))
    if turn_l1 is not None:
        parts.append(_clamp01(turn_l1 / SINGLE_TURN_HIST_L1_TOL))

    if not parts:
        return 0.0
    return float(sum(parts) / len(parts))


def _pair_match_score(template_entity, candidate_feature, *, size_tol, dist_error_ratio=None):
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


def _resolve_score_max(score_max):
    raw = MATCH_SCORE_MAX_DEFAULT if score_max is None else score_max
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return MATCH_SCORE_MAX_DEFAULT
    return _clamp01(value)


def _entity_basic_match(template_entity, candidate_feature, size_tol=None):
    if candidate_feature.get("type") != template_entity.get("type"):
        return False
    tol = size_tol if size_tol is not None else max(SIZE_TOL_MIN, template_entity["size"] * SIZE_TOL_RATIO)
    if abs(candidate_feature.get("size", 0.0) - template_entity["size"]) >= tol:
        return False
    return _single_entity_plugin_pass(template_entity, candidate_feature)


def _single_entity_matches(all_fp, anchor_tmpl):
    return [i for i, feature in enumerate(all_fp) if _entity_basic_match(anchor_tmpl, feature)]


def _pick_anchor(entities, all_fp, type_index, sizes):
    best_idx = 0
    best_count = float("inf")
    for i, entity in enumerate(entities):
        entity_type = entity["type"]
        if entity_type not in type_index:
            return i
        tol = max(SIZE_TOL_MIN, entity["size"] * SIZE_TOL_RATIO)
        base_idx = type_index[entity_type][np.abs(sizes[type_index[entity_type]] - entity["size"]) < tol]
        if len(base_idx) == 0:
            return i
        count = sum(1 for idx in base_idx if _single_entity_plugin_pass(entity, all_fp[int(idx)]))
        if count < best_count:
            best_count = count
            best_idx = i
    return best_idx


def _best_scored_matching(adj_scored):
    n = len(adj_scored)
    if n == 0:
        return set(), 0.0

    right_nodes = sorted({idx for row in adj_scored for idx, _ in row})
    if len(right_nodes) < n:
        return None

    node_map = {rid: j for j, rid in enumerate(right_nodes)}
    inf_cost = 1e6
    cost = np.full((n, len(right_nodes)), inf_cost, dtype=np.float64)
    for i, row in enumerate(adj_scored):
        for rid, score in row:
            j = node_map[rid]
            if score < cost[i, j]:
                cost[i, j] = score

    row_ind, col_ind = linear_sum_assignment(cost)
    if len(row_ind) != n:
        return None
    selected = cost[row_ind, col_ind]
    if np.any(selected >= inf_cost):
        return None

    matched = {right_nodes[int(j)] for j in col_ind}
    return matched, float(np.mean(selected))


def build_session_from_doc(
    doc,
    *,
    fast_build=False,
    drop_noisy_types=True,
    drop_entity_types=None,
    drop_most_common_circle=False,
):
    type_counts_before_drop = _count_entity_types(doc.modelspace())
    selected_types = parse_entity_types(drop_entity_types)
    if not selected_types:
        selected_types = sorted(DEFAULT_DROP_ENTITY_TYPES)

    drop_removed_count = 0
    drop_removed_by_type = {}
    if drop_noisy_types:
        drop_removed_count, drop_removed_by_type = _drop_entities_by_type(doc, selected_types)

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

    insert_scanned, insert_fixed = _sanitize_doc_inserts(doc)
    insert_removed = _drop_unrenderable_inserts(doc)

    msp = doc.modelspace()
    dxf_entity_count = len(msp)
    type_counts_after_drop = _count_entity_types(msp)

    t0 = time.perf_counter()
    session = _build_session_payload(msp, fast_build=fast_build)
    build_time_ms = int((time.perf_counter() - t0) * 1000)

    stats = {
        "status": "success",
        "session_id": session["session_id"],
        "build_mode": "fast" if fast_build else "accurate",
        "drop_noisy_types_enabled": bool(drop_noisy_types),
        "drop_selected_types": selected_types if drop_noisy_types else [],
        "drop_removed_count": int(drop_removed_count),
        "drop_removed_by_type": drop_removed_by_type,
        "entity_type_counts_before_drop": type_counts_before_drop,
        "entity_type_counts_after_drop": type_counts_after_drop,
        "drop_most_common_circle_enabled": bool(circle_plugin_result.get("enabled", False)),
        "drop_most_common_circle_applied": bool(circle_plugin_result.get("applied", False)),
        "drop_most_common_circle_removed": int(circle_plugin_result.get("removed_count", 0)),
        "drop_most_common_circle_radius": circle_plugin_result.get("radius"),
        "drop_most_common_circle_bucket_count": int(circle_plugin_result.get("bucket_count", 0)),
        "drop_most_common_circle_unique_radius_count": int(
            circle_plugin_result.get("unique_radius_count", 0)
        ),
        "drop_most_common_circle_error": circle_plugin_result.get("error"),
        "dxf_entity_count": dxf_entity_count,
        "entity_count": len(session["fingerprints"]),
        "preview_primitive_count": len(session["preview_primitives"]),
        "cache_build_time_ms": build_time_ms,
        "insert_scanned": insert_scanned,
        "insert_fixed": insert_fixed,
        "insert_removed": insert_removed,
        "coord_basis": "dxf_scene",
    }
    return session, stats


def build_session_from_path(
    path,
    *,
    fast_build=False,
    drop_noisy_types=True,
    drop_entity_types=None,
    drop_most_common_circle=False,
):
    try:
        doc = ezdxf.readfile(path)
    except Exception as exc:
        raise ValueError(f"Failed to read DXF: {exc}") from exc
    return build_session_from_doc(
        doc,
        fast_build=fast_build,
        drop_noisy_types=drop_noisy_types,
        drop_entity_types=drop_entity_types,
        drop_most_common_circle=drop_most_common_circle,
    )


def extract_template_from_polygon(session, polygon_vertices):
    if len(polygon_vertices) < 3:
        return {"entities": [], "highlights": []}

    roi_polygon = ShapelyPolygon([(float(x), float(y)) for x, y in polygon_vertices])
    entities_found = [
        feature
        for feature in session["fingerprints"]
        if roi_polygon.covers(Point(feature["x"], feature["y"]))
    ]

    if not entities_found:
        return {"entities": [], "highlights": []}

    group_cx = sum(entity["x"] for entity in entities_found) / len(entities_found)
    group_cy = sum(entity["y"] for entity in entities_found) / len(entities_found)

    entities_clean = [{k: v for k, v in entity.items() if k != "geometry"} for entity in entities_found]
    template_payload = {
        "group_center": {"x": round(group_cx, 3), "y": round(group_cy, 3)},
        "entities": entities_clean,
    }
    template_id = _store_template(session, template_payload["group_center"], entities_clean)
    all_highlights = [entity["geometry"] for entity in entities_found]
    highlights = [_compact_geometry(geometry) for geometry in all_highlights[:MAX_EXTRACT_HIGHLIGHTS_RETURN]]
    session["last_access"] = time.time()

    return {
        "template_id": template_id,
        "group_center": template_payload["group_center"],
        "entity_count": len(entities_clean),
        "entities_preview": entities_clean[:MAX_EXTRACT_ENTITIES_PREVIEW],
        "highlights": highlights,
        "highlight_count_total": len(all_highlights),
        "highlight_count_returned": len(highlights),
        "highlight_truncated": len(all_highlights) > len(highlights),
    }


def extract_template_from_scene_polygon(session, scene_polygon_vertices):
    return extract_template_from_polygon(session, scene_points_to_dxf(scene_polygon_vertices))


def _build_match(
    anchor_idx_or_feat,
    anchor_tmpl,
    group_center,
    used,
    all_fp,
    *,
    max_highlights=MAX_SCAN_HIGHLIGHTS_RETURN,
    match_score=None,
):
    dedupe_key = None
    if isinstance(anchor_idx_or_feat, (int, np.integer)):
        anchor_idx = int(anchor_idx_or_feat)
        anchor_feature = all_fp[anchor_idx]
        dedupe_key = f"anchor:{anchor_idx}"
    else:
        anchor_feature = anchor_idx_or_feat

    cx, cy = anchor_feature["x"], anchor_feature["y"]
    dx = cx - anchor_tmpl["x"]
    dy = cy - anchor_tmpl["y"]
    if group_center:
        match_cx = group_center["x"] + dx
        match_cy = group_center["y"] + dy
    else:
        match_cx, match_cy = cx, cy

    match_features = [anchor_feature] + [all_fp[i] for i in used]
    selected = match_features if max_highlights is None else match_features[:max_highlights]
    highlights = [_compact_geometry(feature["geometry"]) for feature in selected]

    return {
        "dxf_x": round(match_cx, 3),
        "dxf_y": round(match_cy, 3),
        "match_score": (round(float(match_score), 6) if match_score is not None else None),
        "highlights": highlights,
        "highlight_count_total": len(match_features),
        "highlight_count_returned": len(highlights),
        "highlight_truncated": len(highlights) < len(match_features),
        "_dedupe_key": (
            dedupe_key
            if dedupe_key is not None
            else f"xy:{round(float(match_cx), 6)}:{round(float(match_cy), 6)}"
        ),
    }


def _trim_match_highlights(matches, max_with_highlights=MAX_SCAN_MATCHES_WITH_HIGHLIGHTS):
    if len(matches) <= max_with_highlights:
        return matches
    for match in matches[max_with_highlights:]:
        match["highlights"] = []
        match["highlight_count_returned"] = 0
        match["highlight_truncated"] = match.get("highlight_count_total", 0) > 0
    return matches


def _dedupe_and_sort_matches(matches):
    best = {}
    for match in matches:
        key = match.get("_dedupe_key")
        if key is None:
            key = (
                round(float(match.get("dxf_x", 0.0)), 6),
                round(float(match.get("dxf_y", 0.0)), 6),
            )
        prev = best.get(key)
        match_score = match.get("match_score")
        prev_score = None if prev is None else prev.get("match_score")
        if prev is None or (match_score is not None and (prev_score is None or match_score < prev_score)):
            best[key] = match
    out = list(best.values())
    out.sort(key=lambda match: (match.get("match_score") is None, match.get("match_score", 1e9)))
    for match in out:
        match.pop("_dedupe_key", None)
    return out


def _finalize_scan_stats(stats, total_start):
    stats["elapsed_ms"] = int((time.perf_counter() - total_start) * 1000)
    scanned = stats.get("neighbor_sets_scanned", 0)
    total_neighbors = stats.get("neighbor_features_total", 0)
    stats["avg_neighbors_per_anchor"] = round((total_neighbors / scanned) if scanned else 0.0, 2)
    return stats


def _resolve_template_for_scan(session, template_id=None, entities=None, group_center=None):
    if template_id:
        stored = load_template(session, template_id)
        if stored is None:
            raise ValueError("Template not found. Please extract template again.")
        return stored.get("entities", []), stored.get("group_center", None)
    return entities or [], group_center


def scan_session(session, *, template_id=None, entities=None, group_center=None, score_max=None, fast=False):
    total_start = time.perf_counter()
    stats = {
        "engine": "fast" if fast else "standard",
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

    all_fp = session["fingerprints"]
    coords = session["coords"]
    sizes = session["sizes"]
    tree = session["tree"]
    type_index = session["type_index"]

    entities, group_center = _resolve_template_for_scan(
        session,
        template_id=template_id,
        entities=entities,
        group_center=group_center,
    )

    resolved_score_max = _resolve_score_max(score_max)
    stats["score_max"] = resolved_score_max
    stats["template_entity_count"] = len(entities)
    if not entities or tree is None:
        return {
            "match_count": 0,
            "matches": [],
            "scan_stats": _finalize_scan_stats(stats, total_start),
        }

    if len(entities) == 1:
        stats["single_entity_mode"] = True
        anchor_tmpl = entities[0]
        anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)
        stats["candidate_anchor_base"] = sum(
            1
            for feature in all_fp
            if feature["type"] == anchor_tmpl["type"]
            and abs(feature["size"] - anchor_tmpl["size"]) < anchor_size_tol
        )
        t_prefilter = time.perf_counter()
        matched_indices = _single_entity_matches(all_fp, anchor_tmpl)
        stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)
        stats["candidate_anchor_after_plugin"] = len(matched_indices)

        t_matching = time.perf_counter()
        matches_found = []
        for idx in matched_indices:
            feature = all_fp[idx]
            score = _pair_match_score(anchor_tmpl, feature, size_tol=anchor_size_tol)
            if score > resolved_score_max:
                stats["score_reject_count"] += 1
                continue
            matches_found.append(
                _build_match(
                    idx,
                    anchor_tmpl,
                    group_center,
                    set(),
                    all_fp,
                    match_score=score,
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

    t_prefilter = time.perf_counter()
    anchor_idx = _pick_anchor(entities, all_fp, type_index, sizes)
    anchor_tmpl = entities[anchor_idx]
    others = [entities[i] for i in range(len(entities)) if i != anchor_idx]
    anchor_pt = np.array([anchor_tmpl["x"], anchor_tmpl["y"]], dtype=np.float64)
    anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)

    targets = []
    for entity in others:
        dist = calculate_distance(anchor_pt, (entity["x"], entity["y"]))
        targets.append(
            {
                "entity": entity,
                "type": entity["type"],
                "size": entity["size"],
                "size_tol": max(SIZE_TOL_MIN, entity["size"] * SIZE_TOL_RATIO),
                "dist": dist,
                "dist_tol": max(DIST_TOL_MIN, dist * DIST_TOL_RATIO),
            }
        )

    max_tol = max(target["dist_tol"] for target in targets)
    search_r = max(target["dist"] for target in targets) + max_tol

    if fast:
        anchor_type = anchor_tmpl["type"]
        if anchor_type not in type_index:
            stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)
            return {
                "match_count": 0,
                "matches": [],
                "scan_stats": _finalize_scan_stats(stats, total_start),
            }

        anchor_indices = type_index[anchor_type]
        mask = np.abs(sizes[anchor_indices] - anchor_tmpl["size"]) < anchor_size_tol
        potential_base = anchor_indices[mask]
        stats["candidate_anchor_base"] = int(len(potential_base))
        potential = np.array(
            [
                int(idx)
                for idx in potential_base
                if _single_entity_plugin_pass(anchor_tmpl, all_fp[int(idx)])
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

        all_locals = tree.query_ball_point(coords[potential], search_r)
        target_types = list(set(target["type"] for target in targets))
        type_idx_arrays = {name: type_index[name] for name in target_types if name in type_index}
        matches_found = []
        t_matching = time.perf_counter()

        for pi, ai in enumerate(potential):
            cand = coords[ai]
            local_arr = np.array(all_locals[pi], dtype=np.intp)
            if len(local_arr) == 0:
                continue
            local_arr = local_arr[local_arr != ai]
            stats["neighbor_sets_scanned"] += 1
            stats["neighbor_features_total"] += int(len(local_arr))

            local_by_type = {}
            for target_type in target_types:
                if target_type in type_idx_arrays:
                    local_by_type[target_type] = np.intersect1d(local_arr, type_idx_arrays[target_type])
                else:
                    local_by_type[target_type] = np.array([], dtype=np.intp)

            adj_scored = []
            skip = False
            for target in targets:
                candidates = local_by_type.get(target["type"], np.array([], dtype=np.intp))
                if len(candidates) == 0:
                    skip = True
                    break
                size_mask = np.abs(sizes[candidates] - target["size"]) < target["size_tol"]
                candidates = candidates[size_mask]
                if len(candidates) == 0:
                    skip = True
                    break
                candidates = np.array(
                    [
                        int(idx)
                        for idx in candidates
                        if _single_entity_plugin_pass(target["entity"], all_fp[int(idx)])
                    ],
                    dtype=np.intp,
                )
                if len(candidates) == 0:
                    skip = True
                    break
                dists = np.linalg.norm(coords[candidates] - cand, axis=1)
                dist_err = np.abs(dists - target["dist"])
                dist_mask = dist_err < target["dist_tol"]
                valid = candidates[dist_mask]
                if len(valid) == 0:
                    skip = True
                    break
                valid_dist_err = dist_err[dist_mask]
                scored = []
                for valid_idx, dist_error in zip(valid.tolist(), valid_dist_err.tolist()):
                    scored.append(
                        (
                            int(valid_idx),
                            _pair_match_score(
                                target["entity"],
                                all_fp[int(valid_idx)],
                                size_tol=target["size_tol"],
                                dist_error_ratio=(dist_error / target["dist_tol"]),
                            ),
                        )
                    )
                adj_scored.append(scored)

            if skip:
                stats["adjacency_fail_count"] += 1
                continue

            matched = _best_scored_matching(adj_scored)
            if matched is None:
                stats["matching_fail_count"] += 1
                continue

            matched_set, edge_mean_score = matched
            anchor_score = _pair_match_score(anchor_tmpl, all_fp[int(ai)], size_tol=anchor_size_tol)
            total_score = MATCH_SCORE_ANCHOR_WEIGHT * anchor_score + (1.0 - MATCH_SCORE_ANCHOR_WEIGHT) * edge_mean_score
            if total_score > resolved_score_max:
                stats["score_reject_count"] += 1
                continue
            matches_found.append(
                _build_match(
                    int(ai),
                    anchor_tmpl,
                    group_center,
                    matched_set,
                    all_fp,
                    match_score=total_score,
                )
            )

        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)
    else:
        potential_base = [
            (idx, feature)
            for idx, feature in enumerate(all_fp)
            if feature["type"] == anchor_tmpl["type"]
            and abs(feature["size"] - anchor_tmpl["size"]) < anchor_size_tol
        ]
        stats["candidate_anchor_base"] = len(potential_base)
        potential = [
            (idx, feature)
            for idx, feature in potential_base
            if _single_entity_plugin_pass(anchor_tmpl, feature)
        ]
        stats["candidate_anchor_after_plugin"] = len(potential)
        stats["prefilter_ms"] = int((time.perf_counter() - t_prefilter) * 1000)

        matches_found = []
        t_matching = time.perf_counter()
        for anchor_candidate_idx, anchor_candidate in potential:
            candidate_point = (anchor_candidate["x"], anchor_candidate["y"])
            local_set = set(tree.query_ball_point(candidate_point, search_r)) - {anchor_candidate_idx}
            stats["neighbor_sets_scanned"] += 1
            stats["neighbor_features_total"] += len(local_set)

            adj_scored = []
            skip = False
            for target in targets:
                valid = []
                for idx in local_set:
                    feature = all_fp[idx]
                    if not _entity_basic_match(target["entity"], feature, size_tol=target["size_tol"]):
                        continue
                    dist = calculate_distance(candidate_point, (feature["x"], feature["y"]))
                    dist_error = abs(dist - target["dist"])
                    if dist_error < target["dist_tol"]:
                        valid.append(
                            (
                                idx,
                                _pair_match_score(
                                    target["entity"],
                                    feature,
                                    size_tol=target["size_tol"],
                                    dist_error_ratio=(dist_error / target["dist_tol"]),
                                ),
                            )
                        )
                if not valid:
                    skip = True
                    break
                adj_scored.append(valid)

            if skip:
                stats["adjacency_fail_count"] += 1
                continue

            matched = _best_scored_matching(adj_scored)
            if matched is None:
                stats["matching_fail_count"] += 1
                continue

            matched_set, edge_mean_score = matched
            anchor_score = _pair_match_score(anchor_tmpl, anchor_candidate, size_tol=anchor_size_tol)
            total_score = MATCH_SCORE_ANCHOR_WEIGHT * anchor_score + (1.0 - MATCH_SCORE_ANCHOR_WEIGHT) * edge_mean_score
            if total_score > resolved_score_max:
                stats["score_reject_count"] += 1
                continue
            matches_found.append(
                _build_match(
                    anchor_candidate_idx,
                    anchor_tmpl,
                    group_center,
                    matched_set,
                    all_fp,
                    match_score=total_score,
                )
            )

        stats["matching_ms"] = int((time.perf_counter() - t_matching) * 1000)

    stats["raw_match_count"] = len(matches_found)
    unique = _dedupe_and_sort_matches(matches_found)
    stats["unique_match_count"] = len(unique)
    session["last_access"] = time.time()
    return {
        "match_count": len(unique),
        "matches": _trim_match_highlights(unique),
        "scan_stats": _finalize_scan_stats(stats, total_start),
    }


register_single_entity_plugin(_plugin_composite_shape_signature)
register_single_entity_plugin(_plugin_composite_shape_histogram)
register_upload_doc_plugin("drop_most_common_circle_size", _plugin_drop_most_common_circle_size)
