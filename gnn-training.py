"""
CAD GNN Pipeline
================
DXF electronic component detection via entity-level node classification.

Usage:
    python cad_gnn_pipeline.py --mode preprocess
    python cad_gnn_pipeline.py --mode train
    python cad_gnn_pipeline.py --mode inference --input original.dxf
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import ezdxf
from sklearn.neighbors import KDTree
from scipy.spatial import KDTree as SpatialKDTree
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

ENTITY_TYPES = [
    "LINE", "ARC", "CIRCLE", "LWPOLYLINE",
    "TEXT", "INSERT", "SPLINE", "ELLIPSE", "OTHER"
]
TYPE2ID   = {t: i for i, t in enumerate(ENTITY_TYPES)}
NUM_TYPES = len(ENTITY_TYPES)

# Class count is data-driven: derived from len(class_names) at preprocess time
# and propagated through cad_graph.pt -> best_model.pt to inference.
#   background_id = len(class_names)         # last id reserved for background
#   num_classes   = len(class_names) + 1     # components + 1 background


# ══════════════════════════════════════════════════════════════════════════════
# 1. Build label map from JSON annotation file
# ══════════════════════════════════════════════════════════════════════════════

def _merge_duplicate_keys(pairs: list[tuple[str, list]]) -> dict[str, list]:
    """
    object_pairs_hook for json.load that detects duplicate top-level class keys
    and merges their instance lists instead of letting later entries silently
    overwrite earlier ones (the default dict behavior).

    Prints a WARNING per duplicate so the user notices, but does not raise
    so a benign duplicate (same class repeated in two parts of the file)
    keeps working.
    """
    result: dict[str, list] = {}
    for k, v in pairs:
        if k in result:
            print(f"  [WARNING] Duplicate class key '{k}' in annotation — "
                  f"merging {len(v)} instance(s) into existing {len(result[k])}.")
            if isinstance(result[k], list) and isinstance(v, list):
                result[k] = result[k] + v
            else:
                print(f"    [ERROR] Cannot merge non-list values for '{k}'; "
                      f"keeping the first one.")
        else:
            result[k] = v
    return result


def build_label_map(annotation_path: str, class_names: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """
    Parse the JSON annotation file and return two mappings:

    JSON format:
        {
            "ObjectName": [[handleID, handleID, ...], [handleID, ...], ...],
            ...
        }
        Each key is a class name. Each inner list is one component instance
        (a group of entity handles that form one component together).

    Duplicate top-level class keys are merged (with a warning). Without this,
    Python's default json.load would silently drop earlier entries and lose
    annotations.

    Args:
        annotation_path: path to the JSON annotation file
        class_names:     ordered list of class names (index == class_id).
                         Built automatically from JSON keys if not provided.
    Returns:
        handle_to_label : dict mapping handle string -> class_id (0~9)
        name_to_id      : dict mapping class name -> class_id
    """
    import json

    with open(annotation_path, "r") as f:
        annotations = json.load(f, object_pairs_hook=_merge_duplicate_keys)

    # Build name -> id mapping from provided class_names list
    name_to_id = {name: i for i, name in enumerate(class_names)}

    handle_to_label: dict[str, int] = {}
    total = 0

    for class_name, instances in annotations.items():
        if class_name not in name_to_id:
            print(f"  [WARNING] Unknown class '{class_name}' in annotation, skipping.")
            continue
        class_id = name_to_id[class_name]
        for instance_handles in instances:
            for handle in instance_handles:
                handle_to_label[str(handle)] = class_id
                total += 1
        print(f"  {class_name:30s} (class {class_id:02d}): "
              f"{len(instances):5} instances, "
              f"{sum(len(g) for g in instances):7,} handles")

    print(f"  Total labeled handles: {total:,}")
    return handle_to_label, name_to_id


# ══════════════════════════════════════════════════════════════════════════════
# 3. Entity geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_bbox(entity) -> tuple[float, float, float, float]:
    """
    Return bounding box center and dimensions (cx, cy, w, h).
    Values are NOT normalized here; normalization happens in load_dataset.
    """
    t = entity.dxftype()
    try:
        if t == "LINE":
            s, e = entity.dxf.start, entity.dxf.end
            xs = [s.x, e.x]
            ys = [s.y, e.y]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            w  = max(xs) - min(xs) + 1e-6
            h  = max(ys) - min(ys) + 1e-6
            return cx, cy, w, h

        elif t in ("ARC", "CIRCLE"):
            c = entity.dxf.center
            r = entity.dxf.radius
            return c.x, c.y, r * 2, r * 2

        elif t == "LWPOLYLINE":
            pts = list(entity.get_points())
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            cx  = (min(xs) + max(xs)) / 2
            cy  = (min(ys) + max(ys)) / 2
            return cx, cy, max(xs) - min(xs) + 1e-6, max(ys) - min(ys) + 1e-6

        else:
            bbox = entity.bbox()
            if bbox is None:
                return 0.0, 0.0, 1e-6, 1e-6
            cx = (bbox.extmin.x + bbox.extmax.x) / 2
            cy = (bbox.extmin.y + bbox.extmax.y) / 2
            w  = bbox.extmax.x - bbox.extmin.x + 1e-6
            h  = bbox.extmax.y - bbox.extmin.y + 1e-6
            return cx, cy, w, h

    except Exception:
        return 0.0, 0.0, 1e-6, 1e-6


def get_extra(entity) -> tuple[float, float, float]:
    """
    Return type-specific geometric features: (length, angle_deg, radius).
    Returns (0, 0, 0) for entity types where these are not applicable.
    """
    t = entity.dxftype()
    try:
        if t == "LINE":
            s, e   = entity.dxf.start, entity.dxf.end
            dx, dy = e.x - s.x, e.y - s.y
            length = float(np.sqrt(dx**2 + dy**2))
            angle  = float(np.degrees(np.arctan2(dy, dx)) % 180)
            return length, angle, 0.0
        elif t in ("ARC", "CIRCLE"):
            return 0.0, 0.0, float(entity.dxf.radius)
    except Exception:
        pass
    return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Load dataset: one or multiple (DXF + JSON) pairs -> combined graph
# ══════════════════════════════════════════════════════════════════════════════

def _extract_features_from_dxf(
    dxf_path:        str,
    annotation_path: str,
    class_names:     list[str],
    layer2id:        dict[str, int],
    num_layers:      int,
    background_id:   int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Parse one (DXF, JSON) pair into feature matrix, labels, and layer ids.

    Coordinates are normalized per-DXF (each drawing has its own canvas scale).
    This keeps spatial features in [0, 1] regardless of drawing size differences.

    Args:
        dxf_path:        path to DXF file
        annotation_path: path to JSON annotation file
        class_names:     ordered class name list (index == class_id)
        layer2id:        global layer name -> layer id mapping (shared across all DXFs)
        num_layers:      total number of unique layers across all DXFs
    Returns:
        features  : (N, NUM_TYPES+11) float32
        labels    : (N,) int64
        layer_ids : (N,) int64
        matched   : number of entities matched to a label
    """
    handle_to_label, _ = build_label_map(annotation_path, class_names)

    doc          = ezdxf.readfile(dxf_path)
    msp          = doc.modelspace()
    all_entities = list(msp)
    N            = len(all_entities)

    # Per-DXF normalization stats
    raw_cx, raw_cy = [], []
    for e in all_entities:
        cx, cy, _, _ = get_bbox(e)
        raw_cx.append(cx)
        raw_cy.append(cy)

    cx_min   = min(raw_cx); cx_max = max(raw_cx)
    cy_min   = min(raw_cy); cy_max = max(raw_cy)
    canvas_w = cx_max - cx_min + 1e-6
    canvas_h = cy_max - cy_min + 1e-6

    feat_dim  = NUM_TYPES + 11
    features  = np.zeros((N, feat_dim),    dtype=np.float32)
    labels    = np.full(N, background_id,  dtype=np.int64)
    layer_ids = np.zeros(N,                dtype=np.int64)

    matched = 0
    for i, entity in enumerate(all_entities):
        handle = entity.dxf.handle if hasattr(entity.dxf, "handle") else None
        if handle is not None and handle in handle_to_label:
            labels[i] = handle_to_label[handle]
            matched  += 1

        tid = TYPE2ID.get(entity.dxftype(), TYPE2ID["OTHER"])
        features[i, tid] = 1.0

        cx, cy, w, h = get_bbox(entity)
        cx_n = (cx - cx_min) / canvas_w
        cy_n = (cy - cy_min) / canvas_h
        w_n  = w / canvas_w
        h_n  = h / canvas_h
        features[i, NUM_TYPES + 0] = cx_n
        features[i, NUM_TYPES + 1] = cy_n
        features[i, NUM_TYPES + 2] = w_n
        features[i, NUM_TYPES + 3] = h_n
        features[i, NUM_TYPES + 4] = w_n * h_n
        features[i, NUM_TYPES + 5] = w_n / (h_n + 1e-6)

        length, angle, radius = get_extra(entity)
        features[i, NUM_TYPES + 6] = length / (canvas_w + 1e-6)
        features[i, NUM_TYPES + 7] = angle  / 180.0
        features[i, NUM_TYPES + 8] = radius / (canvas_w + 1e-6)

        layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
        lid   = layer2id.get(layer, 0)
        layer_ids[i]               = lid
        features[i, NUM_TYPES + 9] = lid / (num_layers + 1e-6)

        color = getattr(entity.dxf, "color", 7)
        features[i, NUM_TYPES + 10] = (color % 256) / 256.0

    return features, labels, layer_ids, matched


def load_dataset(
    pairs:       list[tuple[str, str]],
    class_names: list[str],
    save_path:   str = "cad_graph.pt",
) -> Data:
    """
    Load multiple (DXF, JSON) pairs and combine them into a single graph.

    Each DXF becomes a disconnected subgraph — edges are only built within
    each DXF (entities from different drawings are never connected).
    All subgraphs share the same node feature space and are concatenated.

    JSON annotation format:
        {
            "ObjectName": [[handleID, handleID, ...], ...],
            ...
        }

    Feature vector per entity (length = NUM_TYPES + 11):
        [0 : NUM_TYPES]      entity type one-hot
        [NUM_TYPES + 0]      cx            (normalized per-DXF)
        [NUM_TYPES + 1]      cy            (normalized per-DXF)
        [NUM_TYPES + 2]      width         (normalized per-DXF)
        [NUM_TYPES + 3]      height        (normalized per-DXF)
        [NUM_TYPES + 4]      area          (normalized per-DXF)
        [NUM_TYPES + 5]      aspect ratio
        [NUM_TYPES + 6]      line length   (normalized per-DXF)
        [NUM_TYPES + 7]      line angle    (normalized 0~1)
        [NUM_TYPES + 8]      radius        (normalized per-DXF)
        [NUM_TYPES + 9]      layer id      (normalized, shared across all DXFs)
        [NUM_TYPES + 10]     color         (normalized)

    Args:
        pairs:       list of (dxf_path, annotation_json_path) tuples
        class_names: ordered list of class names (index == class_id; any length >= 2)
        save_path:   output path for the serialized graph (.pt)
    Returns:
        Combined PyG Data object
    """
    import json

    print(f"\nLoading {len(pairs)} DXF+JSON pair(s)...")

    background_id = len(class_names)        # last id reserved for background
    num_classes   = background_id + 1       # components + 1 background

    # -- Step 1: build global layer encoding across all DXFs --------------
    print("\n[1/4] Scanning all DXFs for layer names...")
    all_layers = set()
    for dxf_path, _ in pairs:
        doc = ezdxf.readfile(dxf_path)
        for e in doc.modelspace():
            if hasattr(e.dxf, "layer"):
                all_layers.add(e.dxf.layer)
    all_layers = sorted(all_layers)
    layer2id   = {layer: i for i, layer in enumerate(all_layers)}
    num_layers = len(layer2id)
    print(f"  Unique layers across all DXFs ({num_layers}): {all_layers}")

    # -- Step 2: extract features from each DXF ---------------------------
    print("\n[2/4] Extracting features from each DXF...")
    all_features:   list[np.ndarray] = []
    all_labels:     list[np.ndarray] = []
    all_layer_ids:  list[np.ndarray] = []
    node_offsets:   list[int]        = [0]   # start index of each DXF in combined graph
    total_matched   = 0
    total_labeled   = 0

    for idx, (dxf_path, ann_path) in enumerate(pairs):
        print(f"  [{idx+1}/{len(pairs)}] {os.path.basename(dxf_path)}")
        features, labels, layer_ids, matched = _extract_features_from_dxf(
            dxf_path, ann_path, class_names, layer2id, num_layers, background_id
        )
        handle_to_label, _ = build_label_map(ann_path, class_names)
        n_labeled    = len(handle_to_label)
        match_rate   = matched / (n_labeled + 1e-6)
        total_matched += matched
        total_labeled += n_labeled

        print(f"    Entities: {len(features):,}  |  "
              f"Match rate: {matched}/{n_labeled} = {match_rate:.1%}")
        if match_rate < 0.90:
            print(f"    [WARNING] Match rate < 90% for {dxf_path}")

        all_features.append(features)
        all_labels.append(labels)
        all_layer_ids.append(layer_ids)
        node_offsets.append(node_offsets[-1] + len(features))

    overall_match = total_matched / (total_labeled + 1e-6)
    print(f"\n  Overall match rate: {total_matched}/{total_labeled} = {overall_match:.1%}")

    # -- Step 3: build per-DXF graphs and combine -------------------------
    print("\n[3/4] Building per-DXF kNN graphs and combining...")
    all_src, all_dst, all_weights = [], [], []

    for idx, (features, layer_ids) in enumerate(zip(all_features, all_layer_ids)):
        offset = node_offsets[idx]
        data_i = build_graph(features, layer_ids, k=16)

        # Shift edge indices by node offset so they index into the combined graph
        src = data_i.edge_index[0].numpy() + offset
        dst = data_i.edge_index[1].numpy() + offset
        all_src.append(src)
        all_dst.append(dst)
        all_weights.append(data_i.edge_attr.numpy().flatten())

    combined_features  = np.concatenate(all_features,  axis=0)
    combined_labels    = np.concatenate(all_labels,    axis=0)
    combined_src       = np.concatenate(all_src,       axis=0)
    combined_dst       = np.concatenate(all_dst,       axis=0)
    combined_weights   = np.concatenate(all_weights,   axis=0)

    data = Data(
        x          = torch.tensor(combined_features, dtype=torch.float),
        edge_index = torch.tensor([combined_src, combined_dst], dtype=torch.long),
        edge_attr  = torch.tensor(combined_weights, dtype=torch.float).unsqueeze(1),
        y          = torch.tensor(combined_labels,  dtype=torch.long),
    )

    # -- Step 4: print summary and save -----------------------------------
    print("\n[4/4] Summary:")
    N = len(combined_labels)
    for c in range(num_classes):
        n    = int((combined_labels == c).sum())
        name = class_names[c] if c < background_id else "background"
        bar  = "=" * min(40, int(n / N * 400))
        print(f"    {name:30s}: {n:>8,}  ({n / N * 100:5.1f}%)  {bar}")

    torch.save({
        "data":         data,
        "layer2id":     layer2id,
        "class_names":  class_names,
        "node_offsets": node_offsets,
        "pairs":        pairs,
    }, save_path)

    print(f"\n  Saved -> {save_path}")
    print(f"  Total nodes : {data.num_nodes:,}")
    print(f"  Total edges : {data.num_edges:,}")
    print(f"  Feature dim : {data.x.shape[1]}")
    print(f"  DXF files   : {len(pairs)}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 5. Graph construction (per-layer kNN)
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(
    features:  np.ndarray,
    layer_ids: np.ndarray,
    k:         int = 16,
) -> Data:
    """
    Build an undirected spatial kNN graph restricted to same-layer edges.

    For each layer, a KDTree is built from entity centers and edges are
    added between each entity and its k nearest neighbors within that layer.
    Edge weight = 1 / (euclidean_distance + eps).

    Args:
        features:  node feature matrix, shape (N, F)
        layer_ids: integer layer id per entity, shape (N,)
        k:         number of nearest neighbors per entity
    Returns:
        PyG Data object with x, edge_index, edge_attr
    """
    centers = features[:, NUM_TYPES:NUM_TYPES + 2].copy()   # (cx, cy)

    src, dst, weights = [], [], []
    unique_layers     = np.unique(layer_ids)

    for lid in unique_layers:
        mask = np.where(layer_ids == lid)[0]
        if len(mask) < 2:
            continue

        k_actual      = min(k, len(mask) - 1)
        layer_centers = centers[mask]

        tree                  = KDTree(layer_centers)
        distances, local_idxs = tree.query(layer_centers, k=k_actual + 1)

        for local_i in range(len(mask)):
            gi = mask[local_i]
            for local_j, dist in zip(local_idxs[local_i][1:],
                                     distances[local_i][1:]):
                gj = mask[local_j]
                w  = 1.0 / (dist + 1e-6)
                # add both directions for undirected graph
                src.append(gi); dst.append(gj); weights.append(w)
                src.append(gj); dst.append(gi); weights.append(w)

    print(f"  Total edges: {len(src):,}")

    return Data(
        x          = torch.tensor(features, dtype=torch.float),
        edge_index = torch.tensor([src, dst], dtype=torch.long),
        edge_attr  = torch.tensor(weights,   dtype=torch.float).unsqueeze(1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Model
# ══════════════════════════════════════════════════════════════════════════════

class CADEntityGNN(nn.Module):
    """
    Graph Attention Network for CAD entity classification.

    Architecture:
        Linear projection -> stacked GATv2 layers (residual + LayerNorm)
        -> MLP classifier head

    Each GATv2 layer uses edge features (spatial distance weights) as
    additional attention bias, letting the model down-weight distant neighbors.
    """

    def __init__(
        self,
        num_classes: int,
        in_dim:      int   = 20,
        hidden:      int   = 256,
        heads:       int   = 8,
        num_layers:  int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels  = hidden,
                out_channels = hidden // heads,
                heads        = heads,
                edge_dim     = 1,
                concat       = True,
                dropout      = dropout,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            x = norm(x + gat(x, edge_index, edge_attr))  # residual + norm
        return self.classifier(x)                         # (N, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Training
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle severe class imbalance.
    Background entities typically dominate by more than 90% of the dataset.
    """
    counts  = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize sum to num_classes
    return weights


def train(
    graph_path:   str   = "cad_graph.pt",
    model_path:   str   = "best_model.pt",
    hidden:       int   = 256,
    num_layers:   int   = 4,
    heads:        int   = 8,
    dropout:      float = 0.1,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs:   int   = 100,
    eval_every:   int   = 5,
):
    """
    Full-batch training on the pre-built graph.

    With 40~172 GB VRAM, 1M nodes fit comfortably in full-batch mode,
    which is simpler and more accurate than mini-batch neighbor sampling.

    The primary metric is foreground accuracy (FG Acc), which measures only
    non-background entities. Overall accuracy is inflated by the majority
    background class and should not be used for model selection.

    Args:
        graph_path:   path to the .pt file produced by load_dataset
        model_path:   output path for the best model checkpoint
        hidden:       hidden dimension for GATv2 layers
        num_layers:   number of GATv2 layers
        heads:        number of attention heads
        dropout:      dropout probability
        lr:           initial learning rate (AdamW)
        weight_decay: L2 regularization coefficient
        num_epochs:   total training epochs
        eval_every:   evaluate and print metrics every N epochs
    """
    print(f"\nLoading graph from {graph_path}...")
    checkpoint    = torch.load(graph_path, map_location="cpu")
    data          = checkpoint["data"]
    layer2id      = checkpoint["layer2id"]
    class_names   = checkpoint["class_names"]
    background_id = len(class_names)
    num_classes   = background_id + 1
    print(f"  Nodes: {data.num_nodes:,}  |  Edges: {data.num_edges:,}")
    print(f"  Classes ({num_classes}): {class_names} + background")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    data = data.to(device)

    class_weights = compute_class_weights(data.y, num_classes).to(device)

    model = CADEntityGNN(
        in_dim      = data.x.shape[1],
        hidden      = hidden,
        heads       = heads,
        num_layers  = num_layers,
        num_classes = num_classes,
        dropout     = dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    best_fg_acc = 0.0
    fg_mask     = data.y != background_id  # foreground mask, reused every epoch

    print(f"\nTraining for {num_epochs} epochs...")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'All Acc':>8}  {'FG Acc':>8}  {'LR':>10}")
    print("-" * 52)

    for epoch in range(1, num_epochs + 1):
        # -- forward + backward --
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr)
        loss   = criterion(logits, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # -- evaluation --
        if epoch % eval_every == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                preds   = logits.argmax(dim=-1)
                all_acc = (preds == data.y).float().mean().item()
                fg_acc  = (preds[fg_mask] == data.y[fg_mask]).float().mean().item()

            current_lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>6}  {loss.item():>8.4f}  "
                  f"{all_acc:>8.3f}  {fg_acc:>8.3f}  {current_lr:>10.2e}")

            if fg_acc > best_fg_acc:
                best_fg_acc = fg_acc
                torch.save({
                    "model_state":  model.state_dict(),
                    "model_config": {
                        "in_dim":      data.x.shape[1],
                        "hidden":      hidden,
                        "heads":       heads,
                        "num_layers":  num_layers,
                        "num_classes": num_classes,
                        "dropout":     dropout,
                    },
                    "layer2id":     layer2id,
                    "class_names":  class_names,
                }, model_path)
                print(f"         ^ Best FG Acc saved -> {model_path}")

    print(f"\nTraining complete. Best FG Acc: {best_fg_acc:.3f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 8. Inference + instance grouping
# ══════════════════════════════════════════════════════════════════════════════


def _build_inference_graph(
    original_path: str,
    graph_path:    str,
    layer2id:      dict[str, int],
    background_id: int,
):
    """
    Build a graph from a DXF file for inference (no labels needed).

    Unlike load_dataset, this requires no class DXF files.
    Features are identical to load_dataset; labels are left as placeholder.

    Args:
        original_path: path to the DXF file to process
        graph_path:    path to save the resulting graph (.pt)
        layer2id:      layer name -> id mapping from training. Layers in this
                       DXF that were not seen during training fall back to id 0.
    Returns:
        (data, meta) tuple
    """
    print(f"\nBuilding inference graph from {original_path}...")
    doc          = ezdxf.readfile(original_path)
    msp          = doc.modelspace()
    all_entities = list(msp)
    N            = len(all_entities)
    num_layers   = len(layer2id)
    print(f"  Total entities: {N:,}")

    raw_cx, raw_cy = [], []
    for e in all_entities:
        cx, cy, _, _ = get_bbox(e)
        raw_cx.append(cx)
        raw_cy.append(cy)

    cx_min, cx_max = min(raw_cx), max(raw_cx)
    cy_min, cy_max = min(raw_cy), max(raw_cy)
    canvas_w = cx_max - cx_min + 1e-6
    canvas_h = cy_max - cy_min + 1e-6

    unknown_layers = sorted({
        e.dxf.layer for e in all_entities
        if hasattr(e.dxf, "layer") and e.dxf.layer not in layer2id
    })
    if unknown_layers:
        print(f"  [WARNING] {len(unknown_layers)} layer(s) unseen in training "
              f"(fall back to id 0): {unknown_layers}")

    feat_dim  = NUM_TYPES + 11
    features  = np.zeros((N, feat_dim), dtype=np.float32)
    layer_ids = np.zeros(N,             dtype=np.int64)

    for i, entity in enumerate(all_entities):
        tid = TYPE2ID.get(entity.dxftype(), TYPE2ID["OTHER"])
        features[i, tid] = 1.0

        cx, cy, w, h = get_bbox(entity)
        cx_n = (cx - cx_min) / canvas_w
        cy_n = (cy - cy_min) / canvas_h
        w_n  = w / canvas_w
        h_n  = h / canvas_h
        features[i, NUM_TYPES + 0] = cx_n
        features[i, NUM_TYPES + 1] = cy_n
        features[i, NUM_TYPES + 2] = w_n
        features[i, NUM_TYPES + 3] = h_n
        features[i, NUM_TYPES + 4] = w_n * h_n
        features[i, NUM_TYPES + 5] = w_n / (h_n + 1e-6)

        length, angle, radius = get_extra(entity)
        features[i, NUM_TYPES + 6] = length / (canvas_w + 1e-6)
        features[i, NUM_TYPES + 7] = angle  / 180.0
        features[i, NUM_TYPES + 8] = radius / (canvas_w + 1e-6)

        layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
        lid   = layer2id.get(layer, 0)
        layer_ids[i]               = lid
        features[i, NUM_TYPES + 9] = lid / (num_layers + 1e-6)

        color = getattr(entity.dxf, "color", 7)
        features[i, NUM_TYPES + 10] = (color % 256) / 256.0

    print("  Building per-layer kNN graph (k=16)...")
    data   = build_graph(features, layer_ids, k=16)
    data.y = torch.full((N,), background_id, dtype=torch.long)

    meta = [
        {
            "type":  e.dxftype(),
            "layer": e.dxf.layer if hasattr(e.dxf, "layer") else "0",
        }
        for e in all_entities
    ]

    torch.save({
        "data":              data,
        "layer2id":          layer2id,
        "canvas":            (cx_min, cy_min, canvas_w, canvas_h),
        "all_entities_meta": meta,
    }, graph_path)
    print(f"  Graph saved -> {graph_path}")
    return data, meta


def inference(
    original_path:  str,
    model_path:     str   = "best_model.pt",
    output_path:    str   = "output_components.dxf",
    conf_threshold: float = 0.7,
    dist_threshold: float = 0.02,
    graph_path:     str   = None,
):
    """
    Run entity classification and group detected entities into component instances.

    The graph is built automatically from the input DXF. Provide graph_path
    to cache the graph and reuse it across multiple inference runs on the
    same file (useful when tuning conf_threshold or dist_threshold).

    Args:
        original_path:  path to the input DXF file
        model_path:     path to the trained model checkpoint
        output_path:    path for the annotated output DXF
        conf_threshold: minimum softmax confidence to accept a prediction
        dist_threshold: normalized spatial distance threshold for grouping
        graph_path:     optional cache path for the built graph (.pt).
                        Defaults to <original_path_stem>_inference_graph.pt.
                        If the file already exists it is reused directly.
    Returns:
        list of dicts: [{class, entity_ids, confidence, size}, ...]
    """
    # -- Derive graph cache path -----------------------------------------
    if graph_path is None:
        base       = os.path.splitext(original_path)[0]
        graph_path = f"{base}_inference_graph.pt"

    # -- Load model (need layer2id + class_names before building graph) --
    print(f"\nLoading model from {model_path}...")
    ckpt          = torch.load(model_path, map_location="cpu")
    config        = ckpt["model_config"]
    layer2id      = ckpt["layer2id"]
    class_names   = ckpt["class_names"]
    background_id = len(class_names)
    model         = CADEntityGNN(**config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Classes ({len(class_names)}): {class_names}")

    # -- Build or load graph ---------------------------------------------
    if os.path.exists(graph_path):
        print(f"\nReusing cached graph: {graph_path}")
        cache = torch.load(graph_path, map_location="cpu")
        data  = cache["data"]
        meta  = cache["all_entities_meta"]
    else:
        data, meta = _build_inference_graph(
            original_path, graph_path, layer2id, background_id
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)
    model  = model.to(device)

    print("Running inference...")
    with torch.no_grad():
        logits     = model(data.x, data.edge_index, data.edge_attr)
        probs      = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_class = probs.argmax(axis=-1)
        pred_conf  = probs.max(axis=-1)

    print(f"  Done. Classifying {len(pred_class):,} entities...")

    # -- Group entities into component instances -------------------------
    features   = data.x.cpu().numpy()
    centers    = features[:, NUM_TYPES:NUM_TYPES + 2]   # (cx, cy)

    valid_mask = (pred_class != background_id) & (pred_conf >= conf_threshold)
    valid_ids  = np.where(valid_mask)[0]
    print(f"  Entities passing threshold (conf >= {conf_threshold}): {len(valid_ids):,}")

    components = []
    for cls in range(background_id):
        cls_ids = valid_ids[pred_class[valid_ids] == cls]
        if len(cls_ids) == 0:
            continue

        cls_centers = centers[cls_ids]
        pairs       = SpatialKDTree(cls_centers).query_pairs(r=dist_threshold)

        n    = len(cls_ids)
        rows = [p[0] for p in pairs] + [p[1] for p in pairs]
        cols = [p[1] for p in pairs] + [p[0] for p in pairs]

        if rows:
            graph_sp = csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(n, n)
            )
        else:
            # No neighboring pairs: each entity forms its own component instance
            graph_sp = csr_matrix((n, n))

        n_comp, comp_labels = connected_components(graph_sp, directed=False)

        for comp_id in range(n_comp):
            local_members  = np.where(comp_labels == comp_id)[0]
            global_members = cls_ids[local_members].tolist()
            components.append({
                "class":      cls,
                "entity_ids": global_members,
                "confidence": float(pred_conf[global_members].mean()),
                "size":       len(global_members),
            })

    print(f"  Total component instances found: {len(components):,}")
    for cls in range(background_id):
        n = sum(1 for c in components if c["class"] == cls)
        print(f"    {class_names[cls]:30s}: {n} instances")

    _write_output_dxf(original_path, components, meta, output_path, class_names)
    print(f"\n  Output saved -> {output_path}")
    return components


def _write_output_dxf(
    original_path: str,
    components:    list[dict],
    meta:          list[dict],
    output_path:   str,
    class_names:   list[str],
):
    """
    Copy detected entities into new annotation layers in the original DXF.

    Each class gets its own layer named DETECTED_<ClassName> with a distinct
    ACI color so detections are immediately visible in any CAD viewer.

    Args:
        original_path: source DXF to annotate
        components:    list of detected component instances
        meta:          entity metadata list (type and layer per entity index)
        output_path:   path to write the annotated DXF
        class_names:   ordered list of class names (index == class_id)
    """
    doc          = ezdxf.readfile(original_path)
    msp          = doc.modelspace()
    all_entities = list(msp)

    # ACI color palette; classes beyond palette length cycle through it
    COLORS = [1, 2, 3, 4, 5, 6, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150, 170, 190, 210]

    def _layer_name(cls: int) -> str:
        # Sanitize class name for DXF layer naming (no spaces / special chars)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in class_names[cls])
        return f"DETECTED_{safe}"

    # Create one detection layer per class
    for cls in range(len(class_names)):
        layer_name = _layer_name(cls)
        if layer_name not in doc.layers:
            doc.layers.new(
                name       = layer_name,
                dxfattribs = {"color": COLORS[cls % len(COLORS)]}
            )

    # Build entity_index -> layer_name map (one pass, avoid duplicate copies)
    eid_to_layer: dict[int, str] = {}
    for comp in components:
        layer_name = _layer_name(comp["class"])
        for eid in comp["entity_ids"]:
            eid_to_layer[eid] = layer_name

    copied = 0
    failed = 0
    for eid, layer_name in eid_to_layer.items():
        if eid >= len(all_entities):
            continue
        src_entity = all_entities[eid]
        try:
            new_entity = src_entity.copy_to_layout(msp)
            new_entity.dxf.layer = layer_name
            copied += 1
        except Exception:
            failed += 1

    print(f"  Entities written to detection layers: {copied:,}  (failed: {failed})")
    doc.saveas(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="CAD GNN Pipeline: DXF electronic component detection"
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "inference"],
        required=True,
        help="Pipeline stage to run"
    )

    # -- preprocess --
    parser.add_argument(
        "--data_dir", default=None,
        help="Directory containing paired DXF and JSON files. "
             "Each pair must share the same stem: drawing.dxf + drawing.json"
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Explicit list of DXF+JSON pairs: dxf1.dxf json1.json dxf2.dxf json2.json ..."
    )
    parser.add_argument(
        "--class_names", nargs="+",
        default=None,
        help="Ordered list of class names matching the JSON keys. "
             "If omitted, inferred (sorted) from the first JSON file."
    )
    parser.add_argument(
        "--graph_path", default="cad_graph.pt",
        help="Output path for the serialized combined graph"
    )

    # -- train --
    parser.add_argument("--model_path",   default="best_model.pt")
    parser.add_argument("--hidden",       type=int,   default=256)
    parser.add_argument("--num_layers",   type=int,   default=4)
    parser.add_argument("--heads",        type=int,   default=8)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=100)

    # -- inference --
    parser.add_argument(
        "--input", default="original.dxf",
        help="Input DXF file for inference (graph is built automatically)"
    )
    parser.add_argument(
        "--output", default="output_components.dxf",
        help="Output DXF path with detection layers"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.7,
        help="Minimum softmax confidence to accept a prediction"
    )
    parser.add_argument(
        "--dist_threshold", type=float, default=0.02,
        help="Normalized spatial distance threshold for instance grouping"
    )
    parser.add_argument(
        "--cache_graph", default=None,
        help="Optional path to cache the built inference graph (.pt). "
             "If the file exists it will be reused, skipping graph construction."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "preprocess":
        import json
        import glob

        # -- Collect (dxf, json) pairs ------------------------------------
        pairs = []

        if args.data_dir:
            # Auto-discover: every .dxf in data_dir paired with same-stem .json
            dxf_files = sorted(glob.glob(os.path.join(args.data_dir, "*.dxf")))
            for dxf_path in dxf_files:
                stem      = os.path.splitext(dxf_path)[0]
                json_path = stem + ".json"
                if os.path.exists(json_path):
                    pairs.append((dxf_path, json_path))
                else:
                    print(f"  [WARNING] No matching JSON for {dxf_path}, skipping.")

        elif args.pairs:
            # Explicit list: dxf1 json1 dxf2 json2 ...
            if len(args.pairs) % 2 != 0:
                raise ValueError("--pairs must have an even number of arguments (dxf json dxf json ...)")
            for i in range(0, len(args.pairs), 2):
                pairs.append((args.pairs[i], args.pairs[i + 1]))

        else:
            raise ValueError("Provide either --data_dir or --pairs for preprocess mode.")

        if not pairs:
            raise ValueError("No valid (DXF, JSON) pairs found.")

        print(f"  Found {len(pairs)} pair(s):")
        for dxf, ann in pairs:
            print(f"    {os.path.basename(dxf):40s}  {os.path.basename(ann)}")

        # -- Resolve class names ------------------------------------------
        if args.class_names is None:
            with open(pairs[0][1], "r") as f:
                ann = json.load(f)
            class_names = sorted(ann.keys())
            print(f"\n  Class names inferred from {os.path.basename(pairs[0][1])} (sorted):")
            for i, name in enumerate(class_names):
                print(f"    {i:2d}: {name}")
        else:
            class_names = args.class_names

        if len(class_names) < 2:
            raise ValueError(
                f"Need at least 2 component classes, got {len(class_names)}: {class_names}"
            )
        print(f"  Total classes: {len(class_names)} components + 1 background")

        load_dataset(
            pairs       = pairs,
            class_names = class_names,
            save_path   = args.graph_path,
        )

    elif args.mode == "train":
        train(
            graph_path   = args.graph_path,
            model_path   = args.model_path,
            hidden       = args.hidden,
            num_layers   = args.num_layers,
            heads        = args.heads,
            dropout      = args.dropout,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            num_epochs   = args.epochs,
        )

    elif args.mode == "inference":
        inference(
            original_path  = args.input,
            model_path     = args.model_path,
            output_path    = args.output,
            conf_threshold = args.conf_threshold,
            dist_threshold = args.dist_threshold,
            graph_path     = args.cache_graph,
        )


if __name__ == "__main__":
    main()
