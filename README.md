# SMDR DXF Template Scanner

FastAPI-based DXF pattern scanner with a full-screen web UI for:

- Uploading DXF files and building a scan cache
- Extracting templates by polygon select or click select
- Removing selected entities from the current cache
- Running standard or fast template matching
- Downloading matched results as DXF
- Downloading matched result handles as JSON
- Saving reusable templates into SQLite-backed libraries

Line-like DXF entities are fingerprinted per source entity. The scanner does not
globally merge neighboring lines before matching, so matched pattern entity counts
stay aligned with the selected template entities.

Block references (`INSERT`) are expanded into selectable virtual entities, and
unsupported drawable DXF entities fall back to a bounding-box fingerprint so SVG
content is less likely to be visible but unselectable.

## Stack

- FastAPI backend in `app.py`
- Single-page frontend in `templates/index.html`
- Template libraries persisted at `data/template_library.sqlite`

## Install

Create a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt fastapi uvicorn
```

## Run

Start the web app:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Main Workflow

1. Upload a DXF file.
2. Optionally filter kept DXF entity types with `Filter types`.
3. Optionally remove the top `N` most common `CIRCLE` radius groups before cache build.
4. Choose template extraction mode:
   - `Polygon`: draw a closed polygon on the canvas
   - `Click`: click an entity directly
5. In `Click` mode:
   - `Click` selects one entity
   - `Shift+Click` adds or removes entities from the selection
6. Run `Scan` with either:
   - `Fast`
   - `Standard`
7. Export matched results with `Download Matched DXF` or `Download Matches JSON`.

## Selection and Editing

- `Clear Selection` clears the current polygon or click selection.
- `Clean Highlight` clears template and scan highlights.
- `Remove Selected` deletes the selected entities from the current uploaded cache only.
- Removed entities are shown as a dark overlay in the viewer.

## Template Library

The UI includes a `Template Library` panel that supports:

- Saving the current selection as a reusable template
- Assigning a category such as `SMD`
- Loading a single saved template
- Loading a category and scanning with all templates in that category
- Deleting the currently loaded saved template
- Deleting a specific saved template from the list
- Creating, renaming, switching, and deleting template libraries
- Deleting an entire category from the active library

### Library Behavior

- Templates are stored in SQLite at `data/template_library.sqlite`.
- All libraries share this single app-wide SQLite file across browser sessions and server restarts.
- Multiple libraries are supported.
- One library is active at a time.
- Existing `data/template_library.json` data is migrated into SQLite on first use.
- If the last library is deleted, the app creates a new empty default library automatically.

## Upload Filters

Before upload, the toolbar supports two preprocessing filters:

- `Filter types`: keep only the DXF types listed in the comma-separated input
- `remove top N circle radii groups`: remove the `N` most frequent `CIRCLE` radius groups

Example:

- `LINE,CIRCLE,LWPOLYLINE`

## Settings

The `Settings` panel is foldable and exposes runtime controls such as:

- Enabled plugins
- Match score thresholds
- Size and distance tolerances
- Server highlight and preview return limits

Pure client display limits such as draw counts and rendered preview caps use fixed
defaults and are not shown in the settings panel.

These settings are sent to the backend and affect extract/scan behavior directly.

## HTTP Endpoints

Core routes currently exposed by the backend:

- `GET /`
- `GET /settings_schema`
- `GET /template_library`
- `POST /template_library/create_library`
- `POST /template_library/rename_library`
- `POST /template_library/select_library`
- `POST /template_library/delete_library`
- `POST /template_library/save`
- `POST /template_library/delete_template`
- `POST /template_library/delete_category`
- `POST /template_library/resolve`
- `POST /upload`
- `POST /extract`
- `POST /template`
- `POST /remove_entities`
- `POST /download_selected_dxf`
- `POST /download_matched_dxf`
- `POST /scan`
- `POST /scan_fast`

## Notes

- The UI defaults to `Fast` scan mode.
- The main canvas fills the full viewer area.
- Template-library responses and the main page are returned with `no-store` cache headers to avoid stale UI after library changes.
