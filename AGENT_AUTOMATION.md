# LLM/VLM CAD Matching Automation

This document describes how an LLM/VLM provider can drive the SMDR matching
pipeline from a user text instruction to classified CAD matches.

The model does not directly produce final matches. It proposes view ROIs and
seed templates. The existing geometry matcher then extracts templates and scans
the DXF cache deterministically.

## Goal

Input CAD drawings are semiconductor package design drawings. A drawing usually
contains:

- top package view
- bottom package view
- title blocks, dimensions, notes, labels, and other explanatory geometry

For each CAD file, the system should find classes such as:

- `SMD`
- `Substrate`
- `Die area`
- `Alignment mark`

The same classes are needed across drawings, but their visual pattern can change
per CAD file. For that reason, automation is adaptive per uploaded CAD: the
model first proposes the relevant ROI and representative seed patterns, then the
matcher finds repeated geometry.

## Architecture

```text
User instruction
  ↓
LLM/VLM proposal
  ↓
views + seed_candidates JSON
  ↓
/agent/run
  ↓
/extract or cached template resolution
  ↓
/scan_fast
  ↓
matches_by_class + validation + highlights
```

The division of responsibility is:

- LLM: parse the user instruction, choose target classes, decide what needs
  review, and output strict JSON.
- VLM: when enabled, inspect a viewer-sized JPG screenshot and propose visual
  ROIs or seed polygons.
- Geometry matcher: extract seed entities and find all matching patterns.
- Human reviewer: initially confirms or corrects model-proposed ROIs/seeds.

## Provider Configuration

Provider setup is entered in the `Agent Review` panel. The provider must expose
an OpenAI-compatible `chat/completions` API.

For Kimi-k2.6 text-only proposal:

```text
Provider Base URL: https://api.moonshot.ai
Provider API Key: <your key>
LLM Model: kimi-k2.6
VLM Model: <leave empty>
Send JPG Screenshot: off
Temperature: 1
Timeout Seconds: 20-60
Max Tokens: 800-1200
```

For a vision-capable provider:

```text
Provider Base URL: <provider base URL>
Provider API Key: <your key if required>
LLM Model: <optional text model>
VLM Model: <vision model>
Send JPG Screenshot: on
Temperature: provider-specific
Timeout Seconds: 30-120
Max Tokens: 1200
```

When `Send JPG Screenshot` is enabled, the browser rasterizes the current CAD
viewer to a screen-sized JPEG and sends that image as `render_image_data_url`.
The full SVG is not sent as an image input.

## Proposal API

`POST /agent/propose`

Request:

```json
{
  "cache_id": "upload-cache-id",
  "instruction": "Find SMD patterns in the top and bottom package views.",
  "target_classes": ["SMD"],
  "views": [],
  "settings": {
    "score_max": 0.4
  },
  "provider": {
    "base_url": "https://api.moonshot.ai",
    "api_key": "...",
    "llm_model": "kimi-k2.6",
    "vlm_model": null,
    "send_image": false,
    "temperature": 1,
    "timeout_seconds": 30,
    "max_tokens": 1200
  }
}
```

If a VLM image is used, the frontend also sends:

```json
{
  "render_image_data_url": "data:image/jpeg;base64,..."
}
```

Expected model output:

```json
{
  "views": [
    {
      "name": "top_view",
      "roi_pct": [[0.08, 0.12], [0.48, 0.82]],
      "confidence": 0.82,
      "source": "vlm"
    },
    {
      "name": "bottom_view",
      "roi_pct": [[0.52, 0.12], [0.92, 0.82]],
      "confidence": 0.8,
      "source": "vlm"
    }
  ],
  "seed_candidates": [
    {
      "target_class": "SMD",
      "view_name": "top_view",
      "label": "small repeated SMD footprint",
      "polygon_pct": [[0.21, 0.34], [0.24, 0.34], [0.24, 0.37], [0.21, 0.37]],
      "confidence": 0.72,
      "source": "vlm"
    }
  ],
  "notes": [
    "Top and bottom view ROIs should be reviewed before final acceptance."
  ]
}
```

All coordinates are render percentage coordinates in `[0, 1]`, not pixels.

## Running the Proposal

The frontend calls `/agent/propose` first. If the response contains seed
candidates, the frontend immediately submits them to `/agent/run`.

`POST /agent/run`

```json
{
  "cache_id": "upload-cache-id",
  "instruction": "Find SMD patterns in the top and bottom package views.",
  "target_classes": ["SMD"],
  "views": [
    {
      "name": "top_view",
      "roi_pct": [[0.08, 0.12], [0.48, 0.82]],
      "confidence": 0.82,
      "source": "vlm"
    }
  ],
  "seed_candidates": [
    {
      "target_class": "SMD",
      "view_name": "top_view",
      "polygon_pct": [[0.21, 0.34], [0.24, 0.34], [0.24, 0.37], [0.21, 0.37]],
      "confidence": 0.72,
      "source": "vlm"
    }
  ],
  "plugins": ["composite_shape_signature"],
  "settings": {
    "score_max": 0.4
  }
}
```

The backend resolves each seed by one of these methods:

- `template_id`: use an existing extracted template in the current upload cache
- `entities` + `group_center`: use inline reviewed entities
- `click_pct`: pick one nearest entity
- `polygon_pct`: extract all entities inside the polygon

Then it calls the fast matcher and returns grouped results.

## Result Shape

`/agent/run` returns:

```json
{
  "status": "completed_with_review",
  "review_required": true,
  "target_classes": ["SMD"],
  "views": [],
  "results": [
    {
      "target_class": "SMD",
      "view_name": "top_view",
      "status": "completed",
      "validation": {
        "confidence": 0.85,
        "warnings": [],
        "entity_count": 4,
        "match_count": 128,
        "mean_match_score": 0.08,
        "score_max": 0.4
      },
      "scan": {
        "match_count": 128,
        "matches": []
      }
    }
  ],
  "matches_by_class": {
    "SMD": {
      "top_view": []
    }
  }
}
```

Validation metadata is used to decide whether the result can be accepted
automatically later.

## Human-in-the-loop to Full Automation

The intended rollout is:

1. Human reviews all model-proposed ROIs and seed candidates.
2. The system records accepted/rejected proposals and match results.
3. High-confidence proposals are auto-accepted.
4. Low-confidence or anomalous results remain in the review queue.

Useful confidence signals include:

- ROI size and placement
- seed `entity_count`
- `match_count`
- match score distribution
- whether matches remain inside the intended ROI
- class-specific expectations

Class-specific guidance:

- `SMD`: best handled by seed extraction plus repeated geometry matching.
- `Alignment mark`: often needs multiple seed templates because mark shapes vary.
- `Die area`: usually a small number of large rectangular regions; may need more
  geometric rules than repeated matching.
- `Substrate`: often the main package outline or board region; template matching
  alone may be insufficient.

## Troubleshooting

`invalid temperature: only 1 is allowed`

Set `Temperature` to `1`. The backend also retries once with `1` for providers
that return this exact error.

`unsupported image format`

Do not send SVG. The current implementation only sends a viewer-sized JPG/PNG
when `Send JPG Screenshot` is enabled. Leave it off for text-only providers.

`Provider request timed out`

Lower `Max Tokens`, increase `Timeout Seconds`, or disable image input. The
error response includes provider debug fields such as timeout, model, and token
budget, but never returns the API key.

Poor ROI or seed proposal

Use the normal click/polygon selector to correct the seed, then click
`Use Current Template as Seed`. This keeps the same `/agent/run` path while
replacing only the model proposal.
