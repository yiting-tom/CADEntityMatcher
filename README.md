# SMDR DXF Pattern Scanner

`SMDR` 是一個以 FastAPI 實作的 DXF 圖樣搜尋工具。流程是：

1. 上傳 DXF 並建立快取
2. 在 SVG 上框選模板
3. 以模板掃描整張圖，找出相似圖樣

目前版本支援：

- Session cache（`cache_id`）
- 模板伺服器端儲存（`template_id`）
- 標準掃描 `/scan` 與快速掃描 `/scan_fast`
- 單一實體與多實體匹配
- 匹配分數（`match_score`）與門檻（`score_max`）
- 掃描統計（`scan_stats`）

---

## 1. 環境需求

- Python 3.10+
- 建議使用虛擬環境

---

## 2. 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install numpy scipy
```

> `app.py` 會用到 `numpy` 與 `scipy`，請確認有安裝。

---

## 3. 啟動服務

```bash
python app.py
```

預設啟動在：

- `http://127.0.0.1:8000`

---

## 4. Web UI 使用流程

打開首頁後照 3 個步驟操作：

1. **Upload DXF File**
- 選擇 `.dxf`
- 可選 `Fast Cache Build (skip ELLIPSE/SPLINE)`
- 點擊 `Upload & Render SVG`

2. **Select Feature (Polygon)**
- 左鍵逐點框選
- 雙擊或點回起點閉合
- 右鍵拖曳平移，滾輪縮放
- 系統會產生 `template_id`

3. **Full Scan (Analyzer)**
- 選 `Standard Scan` 或 `Fast Scan (KD-Tree)`
- 點 `Start Pattern Matching`
- 可下載 fingerprint 與 match results

---

## 5. 核心 API

### `POST /upload`

上傳 DXF，回傳 `cache_id` 與 SVG。

- FormData:
  - `file`: DXF 檔案
  - `fast_build`: `true/false`

回傳重點欄位：

- `cache_id`
- `entity_count`
- `svg_render_time_ms`
- `cache_build_time_ms`

### `POST /extract`

從框選多邊形擷取模板。

請求範例：

```json
{
  "cache_id": "<cache_id>",
  "polygon_pct": [[0.1,0.2],[0.2,0.2],[0.2,0.3],[0.1,0.3]]
}
```

回傳重點欄位：

- `template_id`
- `group_center`
- `entity_count`
- `entities_preview`
- `highlight_count_total` / `highlight_count_returned`

### `POST /template`

用 `cache_id + template_id` 取回完整模板。

### `POST /scan`、`POST /scan_fast`

以模板掃描整張 DXF。

請求建議（template_id 模式）：

```json
{
  "cache_id": "<cache_id>",
  "template_id": "<template_id>",
  "score_max": 0.4
}
```

也支援直接傳完整模板（不建議，payload 較大）：

```json
{
  "cache_id": "<cache_id>",
  "group_center": {"x": 10.0, "y": 20.0},
  "entities": [...]
}
```

回傳重點欄位：

- `match_count`
- `matches[]`
  - `dxf_x`, `dxf_y`
  - `match_score`（越小越好）
  - `highlights`
- `scan_stats`

---

## 6. 參數功能總覽

### API 請求參數

- `cache_id`
  - 來源：`/upload`
  - 作用：指定要使用哪次上傳建立的快取
- `template_id`
  - 來源：`/extract`
  - 作用：指定要使用哪個模板掃描
- `polygon_pct`
  - 作用：框選多邊形，座標採 0~1 百分比（相對 SVG）
- `fast_build`
  - `true`：建 cache 較快（略過 ELLIPSE/SPLINE）
  - `false`：建 cache 較完整
- `score_max`
  - 作用：匹配分數上限，超過即剔除
  - 範圍：`0.0 ~ 1.0`（越小越嚴格）

### 內部可調常數（`app.py`）

- `SIZE_TOL_RATIO`, `SIZE_TOL_MIN`
  - 控制尺寸匹配容差
- `DIST_TOL_RATIO`, `DIST_TOL_MIN`
  - 控制模板內相對距離容差
- `SINGLE_POINT_COUNT_TOL`
  - `COMPOSITE_SHAPE` 點數容差
- `SINGLE_ASPECT_RATIO_TOL`
  - `COMPOSITE_SHAPE` 長寬比容差
- `SINGLE_DIAG_TOL`
  - `COMPOSITE_SHAPE` 對角長容差

---

## 7. `score_max` 調參建議

`score_max` 預設是 `0.4`，範圍 `0.0 ~ 1.0`。

- 誤報太多：調小，例如 `0.2`、`0.1`
- 漏報太多：調大，例如 `0.5`

一般建議：

- 高精準搜尋：`0.1 ~ 0.25`
- 平衡模式：`0.25 ~ 0.4`

---

## 8. `scan_stats` 欄位解讀

- `candidate_anchor_base`
  - 初始候選數（只看 type/size）
- `candidate_anchor_after_plugin`
  - plugin（含 shape signature）過濾後候選數
- `score_reject_count`
  - 被 `score_max` 淘汰的匹配數
- `prefilter_ms`
  - 前置過濾耗時
- `matching_ms`
  - 匹配主流程耗時
- `elapsed_ms`
  - 掃描總耗時
- `avg_neighbors_per_anchor`
  - 每個 anchor 探索的平均鄰居數

---

## 9. 效能與穩定性說明

- 快取是記憶體 Session cache（非永久）
- 前端有高亮繪製上限，避免大量 SVG 導致卡頓
- `scan_stats` 可用來觀察：
  - 候選數量
  - 過濾後數量
  - 匹配耗時
  - `score_reject_count`

---

## 10. 常見問題

### 為什麼有 `cache_id` 失效？

快取是暫存，過期或重啟後需重新上傳 DXF。

### `scan` 和 `scan_fast` 結果要一樣嗎？

理論上應一致；`scan_fast` 主要優化速度。

### 匹配結果太多怎麼辦？

先調小 `score_max`，再縮小模板框選範圍。
