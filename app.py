import os
import math
import itertools
import numpy as np
from scipy.spatial import cKDTree
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import ezdxf
from ezdxf.bbox import extents
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.svg import SVGBackend
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely.ops import linemerge

# Initialize FastAPI app
app = FastAPI()

# Temporary upload path
TEMP_DXF_PATH = "temp_uploaded.dxf"

# --- Algorithm constants ---
SNAP_DECIMALS = 6  # Endpoint rounding precision for stable linemerge stitching
SIZE_TOL_RATIO = 0.05  # Relative size tolerance ratio (5%)
SIZE_TOL_MIN = 0.05  # Minimum absolute size tolerance
DIST_TOL_RATIO = 0.05  # Relative distance tolerance ratio (5%)
DIST_TOL_MIN = 0.1  # Minimum absolute distance tolerance
SVG_LINEWEIGHT_SCALING = 0.2  # Make preview SVG strokes thinner


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


def extract_template_features(msp, roi_box=None):
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
            try:
                pts = [
                    (snap(p.x), snap(p.y)) for p in entity.flattening(0.01)
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
                    "geometry": {
                        "kind": "polyline",
                        "points": [list(c) for c in geom.coords],
                    },
                }
            )

    # --- ROI filtering: keep features whose centroids are inside ROI ---
    if roi_box is not None:
        features = [
            f for f in features if roi_box.contains(Point(f["x"], f["y"]))
        ]

    return features


# --- Global cache (computed once after upload) ---
_cache = {"ready": False}


def _build_cache(msp):
    """Compute and cache all fingerprints and spatial indexes once after upload."""
    bounds = get_dxf_bounds(msp)
    fingerprints = extract_template_features(msp, roi_box=None)

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

    _cache.update(
        {
            "ready": True,
            "bounds": bounds,
            "fingerprints": fingerprints,
            "coords": coords,
            "sizes": sizes,
            "tree": tree,
            "type_index": type_index,
        }
    )


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


def _pick_anchor(entities, type_index, sizes):
    """Pick the entity with the fewest global candidates as anchor (rarest first)."""
    best_idx, best_cnt = 0, float("inf")
    for i, e in enumerate(entities):
        t = e["type"]
        if t not in type_index:
            return i  # Type not found globally -> 0 candidates; fast fail
        tol = max(SIZE_TOL_MIN, e["size"] * SIZE_TOL_RATIO)
        cnt = int(np.sum(np.abs(sizes[type_index[t]] - e["size"]) < tol))
        if cnt < best_cnt:
            best_cnt = cnt
            best_idx = i
    return best_idx


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
    </style>
</head>
<body>
    <h1>PCB CAD Pattern Scanner</h1>

    <div class="step-container">
        <h3>Step 1: Upload DXF File</h3>
        <input type="file" id="dxf-file" accept=".dxf">
        <button onclick="uploadDXF()">Upload &amp; Render SVG</button>
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
        <pre id="extract-result">No data yet</pre>
    </div>

    <div class="step-container">
        <h3>Step 3: Full Scan (Analyzer)</h3>
        <label><input type="radio" name="scan-mode" value="/scan" checked> Standard Scan</label>
        <label style="margin-left:12px;"><input type="radio" name="scan-mode" value="/scan_fast"> Fast Scan (KD-Tree)</label>
        <br><br>
        <button id="btn-scan" onclick="scanTemplate()" disabled>Start Pattern Matching</button>
        <h4 id="scan-status"></h4>
        <pre id="scan-result">Waiting for scan...</pre>
    </div>

    <script>
        // --- DOM ---
        const container = document.getElementById('canvas-container');
        const wrapper   = document.getElementById('transform-wrapper');
        const svgDisplay = document.getElementById('svg-display');
        const svgOverlay = document.getElementById('svg-overlay');
        const polyG = document.getElementById('poly-group');
        const tmplG = document.getElementById('template-group');
        const hlG   = document.getElementById('hl-group');

        // --- State ---
        let scale = 1, panX = 0, panY = 0;
        let isPanning = false, panLastX = 0, panLastY = 0;
        let polyPts = [], polyClosed = false;
        let currentTemplate = null;
        let contentW = 0, contentH = 0;
        let svgViewport = { x0: 0, y0: 0, w: 0, h: 0 };

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

        // --- Upload ---
        async function uploadDXF() {
            var fi = document.getElementById('dxf-file');
            if (!fi.files[0]) return alert("Please select a file");
            var fd = new FormData(); fd.append("file", fi.files[0]);
            document.getElementById('extract-result').innerText = "Uploading and converting to SVG...";
            clearAll();
            var res = await fetch('/upload', { method: 'POST', body: fd });
            var data = await res.json();
            if (data.svg) {
                svgDisplay.innerHTML = data.svg;
                requestAnimationFrame(function() {
                    syncCanvasSize();
                    document.getElementById('extract-result').innerText = "Upload successful. Click to add vertices and select a feature.";
                });
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
            polyClosed = true;
            drawPoly(null);
            var pct = polyPts.map(function(p){ return contentToSvgPct(p); });
            document.getElementById('extract-result').innerText = "Extracting features...";
            var res = await fetch('/extract', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ polygon_pct: pct })
            });
            var result = await res.json();
            document.getElementById('extract-result').innerText = JSON.stringify(result, null, 2);
            if (result.entities && result.entities.length > 0) {
                currentTemplate = result;
                renderTemplateHighlights(result.highlights || []);
                document.getElementById('btn-scan').disabled = false;
            } else {
                tmplG.innerHTML = '';
                document.getElementById('extract-result').innerText += "\\nNo valid entities found in selection.";
                document.getElementById('btn-scan').disabled = true;
            }
        }

        function renderTemplateHighlights(highlights) {
            var NS = "http://www.w3.org/2000/svg";
            var sw = 2 / scale;
            tmplG.innerHTML = "";
            highlights.forEach(function(h) {
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
                    for (var i = 0; i < h.points_pct.length - 1; i++) {
                        var ln = document.createElementNS(NS, "line");
                        var p1 = svgPctToContent(h.points_pct[i][0], h.points_pct[i][1]);
                        var p2 = svgPctToContent(h.points_pct[i+1][0], h.points_pct[i+1][1]);
                        ln.setAttribute("x1", p1.x);
                        ln.setAttribute("y1", p1.y);
                        ln.setAttribute("x2", p2.x);
                        ln.setAttribute("y2", p2.y);
                        ln.setAttribute("stroke", "#00c853");
                        ln.setAttribute("stroke-width", sw);
                        tmplG.appendChild(ln);
                    }
                }
            });
        }

        function clearPoly() {
            polyPts = []; polyClosed = false; polyG.innerHTML = '';
            tmplG.innerHTML = '';
            currentTemplate = null;
            document.getElementById('btn-scan').disabled = true;
            document.getElementById('extract-result').innerText = "Cleared. Click to start a new selection.";
        }
        function resetView() { scale = 1; panX = 0; panY = 0; applyTransform(); }
        function clearAll() {
            clearPoly(); hlG.innerHTML = '';
            resetView();
            document.getElementById('scan-status').innerText = '';
            document.getElementById('scan-result').innerText = 'Waiting for scan...';
        }

        // --- Scan ---
        async function scanTemplate() {
            if (!currentTemplate) return;
            var endpoint = document.querySelector('input[name="scan-mode"]:checked').value;
            document.getElementById('scan-status').innerText = "Scanning...";
            document.getElementById('scan-result').innerText = "";
            hlG.innerHTML = "";
            var t0 = performance.now();
            var res = await fetch(endpoint, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentTemplate)
            });
            var data = await res.json();
            var elapsed = ((performance.now() - t0) / 1000).toFixed(2);
            document.getElementById('scan-status').innerText = 'Scan complete! Found ' + data.match_count + ' matches (' + elapsed + 's)';
            document.getElementById('scan-result').innerText = JSON.stringify(data.matches, null, 2);
            renderHighlights(data.matches);
        }

        function renderHighlights(matches) {
            var NS = "http://www.w3.org/2000/svg";
            var sw = 2 / scale;
            matches.forEach(function(match) {
                if (!match.highlights) return;
                match.highlights.forEach(function(h) {
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
                        for (var i = 0; i < h.points_pct.length - 1; i++) {
                            var ln = document.createElementNS(NS, "line");
                            var p1 = svgPctToContent(h.points_pct[i][0], h.points_pct[i][1]);
                            var p2 = svgPctToContent(h.points_pct[i+1][0], h.points_pct[i+1][1]);
                            ln.setAttribute("x1", p1.x);
                            ln.setAttribute("y1", p1.y);
                            ln.setAttribute("x2", p2.x);
                            ln.setAttribute("y2", p2.y);
                            ln.setAttribute("stroke", "red"); ln.setAttribute("stroke-width", sw);
                            hlG.appendChild(ln);
                        }
                    }
                });
            });
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
async def upload_dxf(file: UploadFile = File(...)):
    with open(TEMP_DXF_PATH, "wb") as buffer:
        buffer.write(await file.read())
    doc = ezdxf.readfile(TEMP_DXF_PATH)
    msp = doc.modelspace()
    backend = SVGBackend()
    render_config = Configuration.defaults().with_changes(
        lineweight_scaling=SVG_LINEWEIGHT_SCALING
    )
    Frontend(RenderContext(doc), backend, config=render_config).draw_layout(msp)
    page = layout.Page(0, 0, layout.Units.mm)
    # Build global cache once (fingerprints + KD-Tree + type index)
    _build_cache(msp)
    return {"status": "success", "svg": backend.get_string(page)}


@app.post("/extract")
async def extract_template(data: dict = Body(...)):
    if not _cache["ready"]:
        return JSONResponse(
            status_code=400, content={"error": "Please upload a file first"}
        )

    polygon_pct = data.get("polygon_pct", [])
    if len(polygon_pct) < 3:
        return {"entities": []}

    bounds = _cache["bounds"]

    # Convert percentage vertices to DXF coordinates
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
        for f in _cache["fingerprints"]
        if roi_polygon.contains(Point(f["x"], f["y"]))
    ]

    if not entities_found:
        return {"entities": []}

    # Compute group centroid
    group_cx = sum(e["x"] for e in entities_found) / len(entities_found)
    group_cy = sum(e["y"] for e in entities_found) / len(entities_found)

    # Remove geometry field to keep template JSON compact
    entities_clean = [
        {k: v for k, v in e.items() if k != "geometry"} for e in entities_found
    ]
    highlights = [geometry_to_render_pct(e["geometry"], bounds) for e in entities_found]

    return {
        "group_center": {"x": round(group_cx, 3), "y": round(group_cy, 3)},
        "entities": entities_clean,
        "highlights": highlights,
    }


def _build_match(anchor_idx_or_feat, anchor_tmpl, group_center, used, bounds, all_fp):
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
    hl = [geometry_to_render_pct(f["geometry"], bounds) for f in mf]

    return {
        "dxf_x": round(mcx, 3),
        "dxf_y": round(mcy, 3),
        "render_pct_x": round(rpx, 4),
        "render_pct_y": round(rpy, 4),
        "highlights": hl,
    }


@app.post("/scan")
async def scan_dxf(template: dict = Body(...)):
    """Standard scan: cache + KD-Tree + bipartite matching (Python loop)."""
    if not _cache["ready"]:
        return JSONResponse(status_code=400, content={"error": "Please upload a file first"})

    bounds = _cache["bounds"]
    all_fp = _cache["fingerprints"]
    tree = _cache["tree"]
    type_index = _cache["type_index"]
    sizes = _cache["sizes"]

    entities = template.get("entities", [])
    group_center = template.get("group_center", None)

    if not entities or tree is None:
        return {"match_count": 0, "matches": []}

    # Single-entity template: search directly by type + size
    if len(entities) == 1:
        anchor_tmpl = entities[0]
        group_center = template.get("group_center", None)
        anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)

        matches_found = [
            _build_match(i, anchor_tmpl, group_center, set(), bounds, all_fp)
            for i, f in enumerate(all_fp)
            if f["type"] == anchor_tmpl["type"]
            and abs(f["size"] - anchor_tmpl["size"]) < anchor_size_tol
        ]
        unique = list(
            {(m["render_pct_x"], m["render_pct_y"]): m for m in matches_found}.values()
        )
        return {"match_count": len(unique), "matches": unique}

    # Pick the rarest entity as anchor
    anchor_idx = _pick_anchor(entities, type_index, sizes)
    anchor_tmpl = entities[anchor_idx]
    others = [entities[i] for i in range(len(entities)) if i != anchor_idx]

    anchor_pt = (anchor_tmpl["x"], anchor_tmpl["y"])
    anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)

    targets = []
    for o in others:
        d = calculate_distance(anchor_pt, (o["x"], o["y"]))
        targets.append(
            {
                "type": o["type"],
                "size": o["size"],
                "size_tol": max(SIZE_TOL_MIN, o["size"] * SIZE_TOL_RATIO),
                "dist": d,
                "dist_tol": max(DIST_TOL_MIN, d * DIST_TOL_RATIO),
            }
        )

    max_tol = max(t["dist_tol"] for t in targets)
    search_r = max(t["dist"] for t in targets) + max_tol

    potential = [
        (i, f)
        for i, f in enumerate(all_fp)
        if f["type"] == anchor_tmpl["type"]
        and abs(f["size"] - anchor_tmpl["size"]) < anchor_size_tol
    ]

    matches_found = []
    for ac_idx, ac in potential:
        cp = (ac["x"], ac["y"])
        local_set = set(tree.query_ball_point(cp, search_r)) - {ac_idx}

        # Build adjacency list
        adj = []
        skip = False
        for t in targets:
            valid = []
            for idx in local_set:
                f = all_fp[idx]
                if f["type"] != t["type"]:
                    continue
                if abs(f["size"] - t["size"]) >= t["size_tol"]:
                    continue
                d = calculate_distance(cp, (f["x"], f["y"]))
                if abs(d - t["dist"]) < t["dist_tol"]:
                    valid.append(idx)
            if not valid:
                skip = True
                break
            adj.append(valid)

        if skip:
            continue

        matched = _find_matching(adj)
        if matched is not None:
            matches_found.append(
                _build_match(ac_idx, anchor_tmpl, group_center, matched, bounds, all_fp)
            )

    unique = list(
        {(m["render_pct_x"], m["render_pct_y"]): m for m in matches_found}.values()
    )
    return {"match_count": len(unique), "matches": unique}


@app.post("/scan_fast")
async def scan_dxf_fast(template: dict = Body(...)):
    """Fast scan: cache + batch KD-Tree + numpy vectorization + bipartite matching."""
    if not _cache["ready"]:
        return JSONResponse(status_code=400, content={"error": "Please upload a file first"})

    bounds = _cache["bounds"]
    all_fp = _cache["fingerprints"]
    coords = _cache["coords"]
    sizes = _cache["sizes"]
    tree = _cache["tree"]
    type_index = _cache["type_index"]

    entities = template.get("entities", [])
    group_center = template.get("group_center", None)

    if not entities or tree is None:
        return {"match_count": 0, "matches": []}

    # Single-entity template: search directly by type + size
    if len(entities) == 1:
        anchor_tmpl = entities[0]
        group_center = template.get("group_center", None)
        a_type = anchor_tmpl["type"]
        if a_type not in type_index:
            return {"match_count": 0, "matches": []}

        anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)
        a_idx = type_index[a_type]
        mask = np.abs(sizes[a_idx] - anchor_tmpl["size"]) < anchor_size_tol
        potential = a_idx[mask]

        matches_found = [
            _build_match(int(ai), anchor_tmpl, group_center, set(), bounds, all_fp)
            for ai in potential
        ]
        unique = list(
            {(m["render_pct_x"], m["render_pct_y"]): m for m in matches_found}.values()
        )
        return {"match_count": len(unique), "matches": unique}

    # Pick the rarest entity as anchor
    anchor_idx = _pick_anchor(entities, type_index, sizes)
    anchor_tmpl = entities[anchor_idx]
    others = [entities[i] for i in range(len(entities)) if i != anchor_idx]

    anchor_pt = np.array([anchor_tmpl["x"], anchor_tmpl["y"]])
    anchor_size_tol = max(SIZE_TOL_MIN, anchor_tmpl["size"] * SIZE_TOL_RATIO)

    targets = []
    for o in others:
        d = float(np.linalg.norm(anchor_pt - [o["x"], o["y"]]))
        targets.append(
            {
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
        return {"match_count": 0, "matches": []}
    a_idx = type_index[a_type]
    mask = np.abs(sizes[a_idx] - anchor_tmpl["size"]) < anchor_size_tol
    potential = a_idx[mask]

    if len(potential) == 0:
        return {"match_count": 0, "matches": []}

    # Batch KD-Tree query: fetch neighbors for all anchors in one call
    all_locals = tree.query_ball_point(coords[potential], search_r)

    # Prebuild type index arrays
    target_types = list(set(t["type"] for t in targets))
    type_idx_arrays = {t: type_index[t] for t in target_types if t in type_index}

    matches_found = []

    for pi, ai in enumerate(potential):
        cand = coords[ai]
        local_arr = np.array(all_locals[pi], dtype=np.intp)
        if len(local_arr) == 0:
            continue

        # Exclude anchor itself
        local_arr = local_arr[local_arr != ai]

        # Pre-intersection: local intersection type_index
        local_by_type = {}
        for tt in target_types:
            if tt in type_idx_arrays:
                local_by_type[tt] = np.intersect1d(local_arr, type_idx_arrays[tt])
            else:
                local_by_type[tt] = np.array([], dtype=np.intp)

        # Build adjacency list (vectorized size + distance filtering)
        adj = []
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
            dists = np.linalg.norm(coords[cands] - cand, axis=1)
            d_mask = np.abs(dists - t["dist"]) < t["dist_tol"]
            valid = cands[d_mask]
            if len(valid) == 0:
                skip = True
                break
            adj.append(valid.tolist())

        if skip:
            continue

        matched = _find_matching(adj)
        if matched is not None:
            matches_found.append(
                _build_match(int(ai), anchor_tmpl, group_center, matched, bounds, all_fp)
            )

    unique = list(
        {(m["render_pct_x"], m["render_pct_y"]): m for m in matches_found}.values()
    )
    return {"match_count": len(unique), "matches": unique}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
