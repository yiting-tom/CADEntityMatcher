# SMDR DXF Pattern Scanner

`SMDR` 是一個以 FastAPI 實作的 DXF 圖樣搜尋工具，主要流程如下：

1. 上傳 DXF 並建立 session cache（`cache_id`）
2. 在 SVG 預覽上框選模板（產生 `template_id`）
3. 對整張圖掃描相似圖樣（`/scan` 或 `/scan_fast`）

本專案的目標是「在可接受速度下提高找樣準確度」，目前內建：

- Session cache（記憶體）
- Template server-side storage（`template_id`）
- 單一實體與多實體匹配
- 標準掃描 `/scan` 與快速掃描 `/scan_fast`
- score-based matching（`match_score`, `score_max`）
- 插件式過濾（單一實體匹配插件、upload 前處理插件）
- Upload/Scan 統計資訊（方便調參）

---

## 1. Environment

- Python 3.10+
- 建議使用虛擬環境
- 主要依賴：`fastapi`, `uvicorn`, `ezdxf`, `shapely`, `numpy`, `scipy`

---

## 2. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install numpy scipy
```

---

## 3. Run

```bash
python app.py
```

預設網址：

- `http://127.0.0.1:8000`

---

## 4. Web UI Quickstart

### Step 1: Upload DXF File

- 選擇 `.dxf` 檔
- 可選參數：
  - `Fast Cache Build (skip ELLIPSE/SPLINE)`：建 cache 更快，特徵較粗
  - `Enable Entity Type Filtering`：刪掉干擾 type（如文字、標註）
  - `Plugin: Drop Most Common CIRCLE Size`：刪除數量最多同半徑 `CIRCLE`
- 點擊 `Upload & Render SVG`

Upload Stats 會顯示：

- entity 總數
- type 統計（before/after drop）
- drop 詳細資訊
- plugin 刪除數量與半徑
- SVG render 時間 / cache 建立時間

### Step 2: Select Feature (Polygon)

- 左鍵：逐點框選
- 雙擊或點回第一點：閉合
- 右鍵拖曳：平移
- 滾輪：縮放
- `ESC`：清除框選
- `Download Fingerprint`：下載模板 JSON

### Step 3: Full Scan (Analyzer)

- 選擇：`Standard Scan` 或 `Fast Scan (KD-Tree)`
- 點擊 `Start Pattern Matching`
- 結果包含：
  - `match_count`
  - `matches[]`
  - `scan_stats`
- `Download Match Results` 可下載完整結果 JSON

---

## 5. API Overview

## `POST /upload`

上傳 DXF、執行前處理、建立 cache、回傳 SVG。

### Request (`multipart/form-data`)

- `file`（required）: DXF 檔案
- `fast_build`（optional, bool）
- `drop_noisy_types`（optional, bool）
- `drop_entity_types`（optional, JSON array 或 CSV）
- `drop_most_common_circle`（optional, bool）

### Response fields (主要)

- `cache_id`: 本次上傳快取 ID
- `entity_count`: 建立 fingerprint 後的特徵數
- `dxf_entity_count`: modelspace entity 數
- `entity_type_counts_before_drop`: 過濾前 type 統計
- `entity_type_counts_after_drop`: 過濾後 type 統計
- `drop_removed_count`: type 過濾刪除數
- `drop_removed_by_type`: type 過濾明細
- `drop_most_common_circle_removed`: circle plugin 刪除數
- `drop_most_common_circle_radius`: 被刪除的目標半徑
- `insert_scanned` / `insert_fixed` / `insert_removed`: INSERT 修復與移除統計
- `svg_render_time_ms`: SVG 渲染時間
- `cache_build_time_ms`: cache 建立時間
- `coord_basis`: 座標映射依據（`render_matrix` / `render_bbox` / `dxf_extents`）
- `svg`: SVG 字串

### cURL example

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@demo.dxf" \
  -F "fast_build=false" \
  -F "drop_noisy_types=true" \
  -F 'drop_entity_types=["TEXT","MTEXT","DIMENSION"]' \
  -F "drop_most_common_circle=true"
```

---

## `POST /extract`

依框選多邊形從 cache 中擷取模板。

### Request JSON

```json
{
  "cache_id": "<cache_id>",
  "polygon_pct": [[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]
}
```

### Response fields (主要)

- `template_id`: 模板 ID
- `group_center`: 模板中心（DXF 座標）
- `entity_count`: 擷取到的特徵數
- `entities_preview`: 模板預覽（截斷）
- `highlights`: 供前端畫框線的幾何
- `highlight_count_total` / `highlight_count_returned`

---

## `POST /template`

用 `cache_id + template_id` 取回模板完整內容。

### Request JSON

```json
{
  "cache_id": "<cache_id>",
  "template_id": "<template_id>"
}
```

---

## `POST /scan` / `POST /scan_fast`

以模板掃描整張 DXF。`scan_fast` 在大圖通常更快。

### 建議 Request（template_id 模式）

```json
{
  "cache_id": "<cache_id>",
  "template_id": "<template_id>",
  "score_max": 0.4
}
```

### 或直接送模板（不建議）

```json
{
  "cache_id": "<cache_id>",
  "group_center": {"x": 10.0, "y": 20.0},
  "entities": [
    {"type": "COMPOSITE_SHAPE", "size": 2.036, "x": 4.239, "y": -1.401}
  ],
  "score_max": 0.35
}
```

### Response fields (主要)

- `match_count`
- `matches[]`
  - `dxf_x`, `dxf_y`
  - `render_pct_x`, `render_pct_y`
  - `match_score`（越小越好）
  - `highlights`
  - `highlight_count_total`, `highlight_count_returned`
- `scan_stats`

---

## 6. Key Parameters

### Upload related

- `fast_build`
  - `true`: 建 cache 快，會略過 `ELLIPSE/SPLINE`
  - `false`: 建 cache 慢一些，但特徵更完整

- `drop_noisy_types`
  - 啟用類型刪除

- `drop_entity_types`
  - 自訂要刪除的 DXF type（JSON array 或 CSV）
  - 若未提供，使用內建預設：
    - `TEXT, MTEXT, DIMENSION, LEADER, MLEADER, HATCH, IMAGE, WIPEOUT, POINT, XLINE, RAY`

- `drop_most_common_circle`
  - 啟用 upload plugin：刪除「數量最多且同半徑」的 circles

### Scan related

- `score_max`
  - 最終匹配分數上限（`0.0 ~ 1.0`）
  - 越小越嚴格，誤報少但可能漏報

---

## 7. Matching Logic (簡化版)

1. 特徵抽取：`CIRCLE` + 線段合併後的 `COMPOSITE_SHAPE`
2. 選擇 anchor（稀有優先）
3. 根據 type/size/distance 找候選
4. 進行 bipartite matching（`linear_sum_assignment`）
5. 計算 `match_score`，超過 `score_max` 剔除
6. 去重 + 排序後回傳

### Single-entity mode

若模板只有 1 個 entity，直接做 type/size + plugin 過濾，不跑完整圖 matching。

---

## 8. Built-in Plugins

## Single Entity Matching Plugins

目前內建兩個 `COMPOSITE_SHAPE` 插件：

1. Signature plugin
- 檢查：`point_count`, `aspect_ratio`, `bbox_diag`

2. Histogram plugin
- 檢查：`edge_hist`（邊長分佈）
- 檢查：`turn_hist`（轉角分佈）
- 用 L1 距離閾值過濾

## Upload Doc Plugin

目前內建：`drop_most_common_circle_size`

- 目標：刪掉同半徑且數量最多的 circles
- 用途：去掉密集重複圓孔、標記圓等高干擾元素

---

## 9. Internal Tunables (`app.py`)

常用可調常數：

- 尺寸與距離容差
  - `SIZE_TOL_RATIO`, `SIZE_TOL_MIN`
  - `DIST_TOL_RATIO`, `DIST_TOL_MIN`

- COMPOSITE_SHAPE 簽名容差
  - `SINGLE_POINT_COUNT_TOL`
  - `SINGLE_ASPECT_RATIO_TOL`
  - `SINGLE_DIAG_TOL`

- Histogram 容差
  - `SINGLE_EDGE_HIST_L1_TOL`
  - `SINGLE_TURN_HIST_L1_TOL`

- Upload circle plugin
  - `CIRCLE_MODE_RADIUS_DECIMALS`
  - `CIRCLE_MODE_MIN_GROUP_COUNT`

- Highlight/回傳上限（避免前端 freeze）
  - `MAX_SCAN_HIGHLIGHTS_RETURN`
  - `MAX_SCAN_MATCHES_WITH_HIGHLIGHTS`
  - `MAX_EXTRACT_HIGHLIGHTS_RETURN`

---

## 10. Tuning Guide (速度 vs 準確度)

如果誤抓很多：

1. 降低 `score_max`（例如 `0.4 -> 0.25`）
2. 開啟 `drop_noisy_types`
3. 開啟 `drop_most_common_circle`
4. 選更小、更具代表性的模板區域

如果漏抓很多：

1. 提高 `score_max`（例如 `0.25 -> 0.4`）
2. 關閉部分 aggressive 過濾
3. 用 `fast_build=false` 建更完整 cache

如果很慢：

1. 改用 `/scan_fast`
2. 開啟 `fast_build`
3. 縮小模板（減少 entity 數）

---

## 11. Troubleshooting

### A. 框到了但 `entities: []`

可能原因：

- 框選區域內沒有 feature centroid
- 當前模板太小或太偏邊界
- 座標映射 fallback 到 `dxf_extents`（精度較差）

建議：

- 重新框稍大範圍
- 看 upload stats 的 `coord_basis`，盡量是 `render_matrix`

### B. `scan_fast` 找不到結果

可能原因：

- 模板被抽成單一 `COMPOSITE_SHAPE`，條件太嚴
- `score_max` 太小
- upload 過濾把關鍵 entity 刪掉

建議：

- 提高 `score_max`
- 先關閉部分過濾
- 改用 `Standard Scan` 比對行為

### C. DWG 轉 DXF 後 render/選取異常

本專案已做：

- INSERT transform sanitize
- render 失敗時移除不可展開 INSERT 後重試
- extract 優先使用 inverse render matrix 映射

若仍有問題，請先用 upload stats 檢查：

- `insert_fixed`
- `insert_removed`
- `coord_basis`

### D. 網頁卡頓或高亮太多

- 已有 highlight 回傳與繪製上限
- 請縮小模板或降低 `score_max`

---

## 12. Cache / Session Behavior

- cache 存在記憶體，不是永久儲存
- `cache_id` 可能因 TTL/容量淘汰失效
- 服務重啟後 cache 全部失效

---

## 13. Developer Notes

### 建議分支策略

- 每個改進開新 branch
- 一個功能一個 commit（或小批次）

### 推薦驗證

```bash
python -m py_compile app.py
```

若有測試檔案，建議再補：

- `/upload` 回傳欄位完整性
- `/extract` 座標映射一致性
- `/scan` 與 `/scan_fast` 結果一致性（允許排序差異）

---

## 14. License

目前未附授權條款，若要公開發佈請補上 `LICENSE`。
