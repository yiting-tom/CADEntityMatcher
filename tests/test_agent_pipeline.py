from pathlib import Path
import importlib.util

from fastapi.testclient import TestClient


BASE_DIR = Path(__file__).resolve().parents[1]
APP_PATH = BASE_DIR / "app.py"
APP_SPEC = importlib.util.spec_from_file_location("smdr_app", APP_PATH)
smdr_app = importlib.util.module_from_spec(APP_SPEC)
APP_SPEC.loader.exec_module(smdr_app)
app = smdr_app.app
_get_cache = smdr_app._get_cache


def _upload_test_dxf(client):
    with (BASE_DIR / "data" / "test.dxf").open("rb") as handle:
        response = client.post(
            "/upload",
            files={"file": ("test.dxf", handle, "application/dxf")},
            data={"fast_build": "true"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["cache_id"]
    return payload


def test_agent_schema_exposes_seed_contract():
    client = TestClient(app)

    response = client.get("/agent/schema")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "experimental"
    assert "SMD" in payload["target_classes"]
    assert "seed_candidates" in payload["request"]


def test_agent_run_without_seed_returns_review_task():
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/run",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns in the package top and bottom views.",
            "target_classes": ["SMD"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_review"
    assert payload["review_required"] is True
    assert payload["views"][0]["name"] == "full_drawing"


def test_agent_render_returns_upload_artifact():
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.get(f"/agent/render/{upload['cache_id']}?include_svg=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cache_id"] == upload["cache_id"]
    assert payload["has_svg"] is True
    assert payload["entity_count"] == upload["entity_count"]
    assert "<svg" in payload["svg"]


def test_agent_propose_without_vllm_returns_config_error(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_LLM_MODEL", raising=False)
    monkeypatch.delenv("VLLM_VLM_MODEL", raising=False)
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/propose",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns.",
            "target_classes": ["SMD"],
        },
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "vLLM is not configured."
    assert payload["fallback"]["status"] == "needs_review"
    assert "provider" in payload


def test_agent_propose_accepts_request_provider_config(monkeypatch):
    captured = {}

    def fake_chat(model, messages, *, temperature=0.0, provider=None):
        captured["model"] = model
        captured["provider"] = provider
        captured["messages"] = messages
        return '{"regions":[]}'

    monkeypatch.setattr(smdr_app, "_vllm_chat_completion", fake_chat)
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/propose",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns.",
            "target_classes": ["SMD"],
            "provider": {
                "base_url": "https://api.example.test",
                "llm_model": "kimi-k2.6",
                "api_key": "secret",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "proposal_ready"
    assert payload["provider"]["base_url"] == "https://api.example.test"
    assert payload["provider"]["api_key_configured"] is True
    assert "api_key" not in payload["provider"]
    assert payload["provider"]["max_tokens"] == 1200
    assert "regions" in payload["proposal"]
    assert captured["model"] == "kimi-k2.6"
    assert captured["provider"].api_key == "secret"


def test_agent_propose_retries_when_provider_requires_temperature_one(monkeypatch):
    calls = []

    def fake_chat(model, messages, *, temperature=0.0, provider=None):
        calls.append(temperature)
        if temperature != 1.0:
            raise RuntimeError(
                'vLLM HTTP 400: {"error":{"message":"invalid temperature: only 1 is allowed for this model"}}'
            )
        return '{"regions":[]}'

    monkeypatch.setattr(smdr_app, "_vllm_chat_completion", fake_chat)
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/propose",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns.",
            "target_classes": ["SMD"],
            "provider": {
                "base_url": "https://api.example.test",
                "llm_model": "kimi-k2.6",
                "api_key": "secret",
            },
        },
    )

    assert response.status_code == 200
    assert calls == [0.0, 1.0]
    assert response.json()["status"] == "proposal_ready"


def test_agent_propose_retries_without_image_when_provider_rejects_svg(monkeypatch):
    message_shapes = []

    def fake_chat(model, messages, *, temperature=0.0, provider=None):
        has_image = any(
            isinstance(msg.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in msg.get("content")
            )
            for msg in messages
        )
        message_shapes.append(has_image)
        if has_image:
            raise RuntimeError(
                'vLLM HTTP 400: {"error":{"message":"Invalid request: unsupported image format: text/xml; charset=utf-8"}}'
            )
        return '{"seeds":[]}'

    monkeypatch.setattr(smdr_app, "_vllm_chat_completion", fake_chat)
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/propose",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns.",
            "target_classes": ["SMD"],
            "settings": {"tile_grid_per_view": 1, "tile_px_long_side": 128},
            "provider": {
                "base_url": "https://api.example.test",
                "vlm_model": "vision-model",
                "api_key": "secret",
                "send_image": True,
            },
        },
    )

    assert response.status_code == 200
    assert message_shapes[0] is True
    assert message_shapes[1] is False
    assert response.json()["status"] == "proposal_ready"


def test_agent_propose_returns_json_when_provider_times_out(monkeypatch):
    def fake_urlopen(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(smdr_app.urllib.request, "urlopen", fake_urlopen)
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/propose",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns.",
            "target_classes": ["SMD"],
            "provider": {
                "base_url": "https://api.example.test",
                "llm_model": "kimi-k2.6",
                "api_key": "secret",
                "timeout_seconds": 5,
                "max_tokens": 512,
            },
        },
    )

    assert response.status_code == 502
    payload = response.json()
    assert "timed out" in payload["error"]
    assert payload["provider"]["timeout_seconds"] == 5.0
    assert payload["provider"]["max_tokens"] == 512


def test_compute_density_grid_buckets_fingerprints():
    bounds = smdr_app._make_bounds(0.0, 0.0, 100.0, 100.0)
    cache_payload = {
        "bounds": bounds,
        "fingerprints": [
            {"type": "CIRCLE", "x": 10.0, "y": 90.0, "size": 1.0},
            {"type": "CIRCLE", "x": 12.0, "y": 88.0, "size": 1.0},
            {"type": "LINE", "x": 80.0, "y": 20.0, "size": 1.0},
        ],
    }
    density = smdr_app._compute_density_grid(cache_payload, gx=10)
    grid = density["grid"]
    assert grid.shape == (10, 10)
    assert grid.sum() == 3
    # x=10 -> column 1, render-y=100-90=10 -> row 1
    assert grid[1, 1] >= 1
    # x=80 -> col 8, render-y=100-20=80 -> row 8
    assert grid[8, 8] == 1


def test_label_connected_regions_separates_two_blobs():
    bounds = smdr_app._make_bounds(0.0, 0.0, 100.0, 100.0)
    fps = []
    for x in range(5, 30, 2):
        for y in range(70, 95, 2):
            fps.append({"type": "CIRCLE", "x": float(x), "y": float(y), "size": 0.5})
    for x in range(60, 90, 2):
        for y in range(10, 35, 2):
            fps.append({"type": "LINE", "x": float(x), "y": float(y), "size": 1.0})
    cache_payload = {"bounds": bounds, "fingerprints": fps}
    density = smdr_app._compute_density_grid(cache_payload, gx=20)
    regions = smdr_app._label_connected_regions(
        density, min_cells=4, close_kernel=1
    )
    assert len(regions) == 2
    assert all("roi_pct" in r for r in regions)


def test_classify_region_heuristic_table_vs_view():
    # Conservative heuristic: line-dominated regions stay 'view' unless
    # there is positive table evidence (high text share). Chip drawings often
    # have NO table at all, so absence of pads must not flip a region to table.
    line_only_features = {
        "entity_count": 200,
        "type_dist": {"COMPOSITE_SHAPE": 0.97, "MTEXT": 0.03},
        "structural_share": 0.97,
        "circle_share": 0.0,
        "text_share": 0.03,
        "dimension_share": 0.0,
        "composite_share": 0.97,
        "size_mean": 1.0,
        "size_std": 0.5,
        "aspect_ratio": 2.5,
        "area_ratio": 0.05,
        "edge_proximity": 0.2,
        "repetition_entropy": 0.85,
    }
    line_only_verdict = smdr_app._classify_region_heuristic(line_only_features)
    assert line_only_verdict["label"] == "view"

    table_features = dict(line_only_features)
    table_features.update({"text_share": 0.55, "structural_share": 0.45,
                           "composite_share": 0.45, "type_dist": {"MTEXT": 0.55, "COMPOSITE_SHAPE": 0.45}})
    table_verdict = smdr_app._classify_region_heuristic(table_features)
    assert table_verdict["label"] == "table"

    view_features = {
        "entity_count": 500,
        "type_dist": {"CIRCLE": 0.6, "COMPOSITE_SHAPE": 0.4},
        "structural_share": 0.4,
        "circle_share": 0.6,
        "text_share": 0.0,
        "dimension_share": 0.0,
        "composite_share": 0.4,
        "size_mean": 0.3,
        "size_std": 0.05,
        "aspect_ratio": 1.2,
        "area_ratio": 0.18,
        "edge_proximity": 0.15,
        "repetition_entropy": 0.3,
    }
    view_verdict = smdr_app._classify_region_heuristic(view_features)
    assert view_verdict["label"] == "view"

    title_features = dict(table_features)
    title_features.update(
        {
            "text_share": 0.0,
            "structural_share": 0.95,
            "composite_share": 0.95,
            "aspect_ratio": 8.0,
            "edge_proximity": 0.01,
            "area_ratio": 0.05,
        }
    )
    title_verdict = smdr_app._classify_region_heuristic(title_features)
    assert title_verdict["label"] == "title_block"

    dimension_features = {
        "entity_count": 60,
        "type_dist": {"DIMENSION": 0.7, "LEADER": 0.3},
        "structural_share": 0.0,
        "circle_share": 0.0,
        "text_share": 0.0,
        "dimension_share": 1.0,
        "composite_share": 0.0,
        "size_mean": 1.0,
        "size_std": 0.2,
        "aspect_ratio": 12.0,
        "area_ratio": 0.04,
        "edge_proximity": 0.01,
        "repetition_entropy": 0.5,
    }
    dimension_verdict = smdr_app._classify_region_heuristic(dimension_features)
    assert dimension_verdict["label"] == "dimension"


def test_agent_regions_endpoint_returns_segmentation():
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.get(f"/agent/regions/{upload['cache_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cache_id"] == upload["cache_id"]
    assert isinstance(payload["regions"], list)
    assert payload["summary"]["entity_total"] >= 0
    for region in payload["regions"]:
        assert region["label"] in {
            "view",
            "detail",
            "table",
            "title_block",
            "dimension",
            "note",
            "unknown",
        }
        assert "features" in region
        assert "label_confidence" in region


def test_classify_regions_llm_falls_back_silently_on_provider_error(monkeypatch):
    def fake_chat(model, messages, *, temperature=0.0, provider=None):
        raise RuntimeError("vLLM connection failed")

    monkeypatch.setattr(smdr_app, "_vllm_chat_completion", fake_chat)
    region = {
        "id": "region_00",
        "roi_pct": [[0.0, 0.0], [0.5, 0.5]],
        "cell_count": 12,
        "entity_count": 100,
        "features": {
            "entity_count": 100,
            "structural_share": 0.55,
            "circle_share": 0.4,
            "text_share": 0.0,
            "dimension_share": 0.0,
            "composite_share": 0.55,
            "size_mean": 0.3,
            "size_std": 0.05,
            "aspect_ratio": 1.2,
            "area_ratio": 0.18,
            "edge_proximity": 0.15,
            "repetition_entropy": 0.4,
            "type_dist": {"CIRCLE": 0.4, "COMPOSITE_SHAPE": 0.55, "MTEXT": 0.05},
        },
        "label": "view",
        "label_confidence": 0.7,
        "label_reasons": ["default_view"],
        "label_source": "heuristic",
        "included_in_pipeline": True,
    }

    class _Provider:
        base_url = "https://api.example.test"
        api_key = "secret"
        llm_model = "kimi-k2.6"
        vlm_model = None
        send_image = False
        temperature = 1
        timeout_seconds = 30
        max_tokens = 512

    refined = smdr_app._classify_regions_llm(
        [region],
        instruction="Find SMD",
        target_classes=["SMD"],
        provider=_Provider(),
    )
    assert refined[0]["label"] == "view"
    assert refined[0]["label_source"] == "heuristic"


def test_iter_tiles_for_view_grid_layout():
    tiles = list(smdr_app._iter_tiles_for_view([[0.2, 0.2], [0.8, 0.8]], grid_x=2))
    assert len(tiles) == 4
    indexes = [t[0] for t in tiles]
    assert indexes == [0, 1, 2, 3]
    rois = [t[1] for t in tiles]
    assert rois[0] == [[0.2, 0.2], [0.5, 0.5]]
    assert rois[3] == [[0.5, 0.5], [0.8, 0.8]]


def test_tile_local_to_full_pct_round_trip():
    tile_origin = (0.2, 0.2)
    tile_size = (0.3, 0.3)
    full = smdr_app._tile_local_to_full_pct([0.5, 0.5], tile_origin, tile_size)
    assert full == [0.35, 0.35]


def test_dedup_seeds_by_geometry_keeps_higher_confidence():
    seeds = [
        {"target_class": "SMD", "click_pct": [0.40, 0.40], "confidence": 0.8},
        {"target_class": "SMD", "click_pct": [0.401, 0.401], "confidence": 0.5},
        {"target_class": "SMD", "click_pct": [0.60, 0.60], "confidence": 0.6},
        {"target_class": "Substrate", "click_pct": [0.40, 0.40], "confidence": 0.7},
    ]
    deduped = smdr_app._dedup_seeds_by_geometry(seeds, dist_threshold=0.01)
    assert len(deduped) == 3
    smd_seeds = [s for s in deduped if s["target_class"] == "SMD"]
    assert len(smd_seeds) == 2
    near_seed = next(s for s in smd_seeds if abs(s["click_pct"][0] - 0.40) < 0.01)
    assert near_seed["confidence"] == 0.8


def test_render_tile_png_returns_png_bytes():
    bounds = smdr_app._make_bounds(0.0, 0.0, 100.0, 100.0)
    cache_payload = {
        "bounds": bounds,
        "fingerprints": [
            {
                "type": "CIRCLE",
                "x": 10.0,
                "y": 10.0,
                "size": 1.0,
                "geometry": {"kind": "circle", "cx": 10.0, "cy": 10.0, "r": 1.0},
            },
            {
                "type": "COMPOSITE_SHAPE",
                "x": 50.0,
                "y": 50.0,
                "size": 5.0,
                "geometry": {
                    "kind": "polyline",
                    "points": [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0]],
                },
            },
        ],
    }
    png = smdr_app._render_tile_png(
        cache_payload, [[0.0, 0.0], [1.0, 1.0]], px_long_side=128
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_agent_tile_endpoint_returns_png():
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.get(
        f"/agent/tile/{upload['cache_id']}",
        params={"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.4, "px": 128},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_agent_propose_tiles_views_and_aggregates_seeds(monkeypatch):
    tile_calls = {"region_classifier": 0, "seed_calls": 0}

    def fake_chat(model, messages, *, temperature=0.0, provider=None):
        text_payload = ""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                text_payload += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_payload += part.get("text", "")
        if "region_00" in text_payload or "Allowed labels: view" in text_payload:
            tile_calls["region_classifier"] += 1
            return '{"regions":[]}'
        tile_calls["seed_calls"] += 1
        return (
            '{"seeds":[{"target_class":"SMD","click_pct":[0.5,0.5],"confidence":0.6}]}'
        )

    monkeypatch.setattr(smdr_app, "_vllm_chat_completion", fake_chat)
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/propose",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns.",
            "target_classes": ["SMD"],
            "settings": {"tile_grid_per_view": 2, "tile_px_long_side": 128},
            "provider": {
                "base_url": "https://api.example.test",
                "llm_model": "kimi-k2.6",
                "vlm_model": "vision-model",
                "api_key": "secret",
                "send_image": True,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "proposal_ready"
    proposal = payload["proposal"]
    assert tile_calls["region_classifier"] == 1
    assert tile_calls["seed_calls"] == 4  # 2x2 tiles for one view region
    assert len(proposal["tile_debug"]) == 4
    assert proposal["seed_candidates"], "expected at least one seed after dedup"
    seed = proposal["seed_candidates"][0]
    assert seed["target_class"] == "SMD"
    assert "click_pct" in seed


def test_agent_propose_marks_text_only_when_vlm_rejects_image(monkeypatch):
    """When the VLM model field points to a text-only model (e.g. kimi-k2.6),
    image-input rejection must transparently retry text-only and downgrade
    seed confidence + source so the human reviewer knows it's unreliable."""
    seen_image_flags = []

    def fake_chat(model, messages, *, temperature=0.0, provider=None):
        has_image = any(
            isinstance(msg.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in msg.get("content")
            )
            for msg in messages
        )
        seen_image_flags.append(has_image)
        text_payload = ""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                text_payload += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_payload += part.get("text", "")
        is_region = "Allowed labels: view" in text_payload
        if is_region:
            return '{"regions":[]}'
        if has_image:
            raise RuntimeError(
                'vLLM HTTP 400: {"error":{"message":"This model does not support image_url input."}}'
            )
        return '{"seeds":[{"target_class":"SMD","click_pct":[0.5,0.5],"confidence":0.7}]}'

    monkeypatch.setattr(smdr_app, "_vllm_chat_completion", fake_chat)
    client = TestClient(app)
    upload = _upload_test_dxf(client)

    response = client.post(
        "/agent/propose",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find SMD patterns.",
            "target_classes": ["SMD"],
            "settings": {"tile_grid_per_view": 1, "tile_px_long_side": 128},
            "provider": {
                "base_url": "https://api.moonshot.ai",
                "llm_model": "kimi-k2.6",
                "vlm_model": "kimi-k2.6",
                "api_key": "secret",
                "temperature": 1,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    proposal = payload["proposal"]
    assert proposal["tile_debug"][0]["status"] == "ok_text_only"
    assert proposal["seed_candidates"], "expected at least one downgraded seed"
    seed = proposal["seed_candidates"][0]
    assert seed["source"] == "llm_tile_text_only"
    assert seed["confidence"] <= 0.3
    assert any(flag for flag in seen_image_flags), "VLM call with image should have been attempted"
    assert any(not flag for flag in seen_image_flags), "fallback text-only retry should have happened"


def test_agent_run_accepts_inline_entity_seed():
    client = TestClient(app)
    upload = _upload_test_dxf(client)
    cache = _get_cache(upload["cache_id"])
    seed_entity = {
        key: value
        for key, value in cache["fingerprints"][0].items()
        if key != "geometry"
    }

    response = client.post(
        "/agent/run",
        json={
            "cache_id": upload["cache_id"],
            "instruction": "Find all matches like this reviewed SMD seed.",
            "target_classes": ["SMD"],
            "views": [
                {
                    "name": "top_view",
                    "roi_pct": [[0.0, 0.0], [1.0, 1.0]],
                    "confidence": 1.0,
                    "source": "test",
                }
            ],
            "seed_candidates": [
                {
                    "target_class": "SMD",
                    "view_name": "top_view",
                    "entities": [seed_entity],
                    "confidence": 1.0,
                    "source": "human_review",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed_with_review"
    result = payload["results"][0]
    assert result["status"] == "completed"
    assert result["validation"]["entity_count"] == 1
    assert "SMD" in payload["matches_by_class"]
