# SMDR Qt DXF Pattern Scanner

`SMDR` 已改成 Qt 桌面版，不再依賴 FastAPI 或 SVG 預覽。

核心調整：

- `app.py` 現在是 Qt 桌面入口
- `smdr_core.py` 負責 DXF 前處理、特徵擷取、模板儲存與掃描
- 框選改成直接在 Qt `QGraphicsScene` 上操作，使用 DXF 幾何對應的 scene 座標
- 不再用 HTML/SVG 百分比與 `viewBox` 反推 DXF，因此 feature select 不會受 `preserveAspectRatio` 或瀏覽器縮放影響

## Coordinate Model

畫面上的 scene 座標採用：

- `scene_x = dxf_x`
- `scene_y = -dxf_y`

這是唯一的顯示轉換，沒有額外百分比正規化。框選完成後會直接把 scene polygon 還原成 DXF polygon，再用 DXF 座標做模板擷取。

對應函式：

- `dxf_to_scene_point()`
- `scene_to_dxf_point()`
- `extract_template_from_scene_polygon()`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Workflow

1. 載入 DXF
2. 在畫布上以 polygon 框選模板
3. 執行 `Standard Scan` 或 `Fast Scan`
4. 匯出模板 JSON 或 match results JSON

## Tests

```bash
python -m pytest tests/test_smdr_core.py
```

測試重點：

- DXF <-> scene 座標 round-trip 一致
- scene polygon 框選後擷取到的 feature 座標與 DXF 完全對應
- `Standard Scan` 與 `Fast Scan` 對同一模板回傳相同 match centers

## Notes

- 目前 GUI 需要 `PySide6`
- 如果環境尚未安裝 Qt，`python app.py` 會直接提示缺少 `PySide6`
