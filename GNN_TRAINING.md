# CAD GNN Pipeline (`gnn-training.py`)

用 **Graph Attention Network** 對 DXF 檔裡的每個實體 (entity) 做分類,
辨識出 N 種(由標註自動決定)電子元件 + 1 個背景類別,並把同類別、
空間相鄰的實體聚成「元件實例」標記回 DXF 圖層。

整條 pipeline 是單一 Python 檔,三個獨立階段:

```
preprocess  →  train  →  inference
 (建圖)        (訓練)      (推論+輸出)
```

---

## 適用情境

- 你有一批已標註好的 DXF 圖(電路圖、PCB 平面圖等)。
- 標註是「**哪些 entity handle 屬於哪個元件**」的群組。
- 想訓練一個模型,套到沒看過的 DXF 上自動找出元件位置。

如果你只是想用既有 template 比對 DXF,請看根目錄的 `README.md`(那是另一條互動式工具線)。

---

## 安裝需求

Python 3.10+,主要相依:

```bash
pip install torch torch-geometric ezdxf scikit-learn scipy numpy
```

`torch-geometric` 的 wheel 與 PyTorch / CUDA 版本綁定,
若安裝失敗請參考 [官方安裝表](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

GPU 不是必要,但 1M+ entity 的圖建議用 GPU 跑(訓練是 full-batch)。

---

## 資料格式

每個訓練樣本是一對檔案,**檔名相同、副檔名不同**:

```
data/train/
├── drawing_001.dxf
├── drawing_001.json
├── drawing_002.dxf
├── drawing_002.json
└── ...
```

### JSON 標註格式

```json
{
  "Resistor":  [["1A2", "1A3", "1A4"], ["2B1", "2B2"]],
  "Capacitor": [["3C7", "3C8", "3C9", "3CA"]],
  "Diode":     [["4D1", "4D2"]]
}
```

- **Key** = 類別名稱(自由命名)
- **Value** = 一個 list of list,外層每個元素代表「一個元件實例」,
  內層 list 是組成這個元件的所有 DXF entity handle(字串)
- **同一類別請只寫一個 key,所有 instance 放在 list 裡**(下方範例的 `Resistor`
  示範兩顆電阻)。Preprocess 偵測到重複 key 會印 `[WARNING]` 並自動合併,
  資料不會掉,但警告本身是要避免的——重複 key 是 JSON 反模式
- **類別數量自由**(至少 2 類)。模型輸出維度 = N + 1(N 個元件類別 + 1 背景)
  從 preprocess 階段的 JSON 自動推得,沿著 `cad_graph.pt` → `best_model.pt`
  傳到 inference,**訓練/推論一定會用同一份 `class_names`**
- 沒被列在 JSON 裡的 entity 會自動標成背景

### Handle 怎麼來?

DXF 裡每個 entity 有唯一 handle(16 進位字串)。兩種拿法:

1. 用 `ezdxf` 寫個小腳本 dump 出來:
   ```python
   import ezdxf, json
   doc = ezdxf.readfile("drawing.dxf")
   print(json.dumps([
       {"handle": e.dxf.handle, "type": e.dxftype(), "layer": e.dxf.layer}
       for e in doc.modelspace()
   ], indent=2))
   ```
2. 在主線的 SMDR scanner UI 裡選取後匯出 JSON。

---

## 三階段使用

### 1. Preprocess — 把 DXF + JSON 變成圖

從資料夾自動配對:

```bash
python gnn-training.py --mode preprocess \
    --data_dir data/train \
    --graph_path cad_graph.pt
```

或顯式指定多組 pair:

```bash
python gnn-training.py --mode preprocess \
    --pairs a.dxf a.json b.dxf b.json \
    --graph_path cad_graph.pt
```

類別名稱與**順序**預設從第一個 JSON 的 key 排序推得。若 dataset 裡某些
類別在第一個 JSON 沒出現,或想固定 detection 圖層順序,用 `--class_names`
顯式指定(順序就是 class id 0, 1, 2, …):

```bash
    --class_names Resistor Capacitor Diode    # 任意數量,至少 2 類
```

產物:`cad_graph.pt`(包含合併後的圖、layer 編碼、`class_names` 等)。

### 2. Train — 訓練 GNN

```bash
python gnn-training.py --mode train \
    --graph_path cad_graph.pt \
    --model_path best_model.pt \
    --epochs 100
```

可調超參數(預設值已標示):

| 參數 | 預設 | 說明 |
|------|------|------|
| `--hidden` | 256 | GAT 隱藏維度 |
| `--num_layers` | 4 | GAT 層數 |
| `--heads` | 8 | attention head 數 |
| `--dropout` | 0.1 | dropout 機率 |
| `--lr` | 1e-3 | 初始學習率 (AdamW) |
| `--weight_decay` | 1e-4 | L2 正則 |
| `--epochs` | 100 | 訓練輪數 |

訓練時 console 會印兩個準確率:

- **All Acc** — 含背景的整體準確率(背景太多容易虛高,不要看)
- **FG Acc** — 只算非背景 entity 的準確率(**模型存檔以這個為準**)

訓練每個 `eval_every`(預設 5)epoch 評估一次,**FG Acc 創新高才存檔**。

### 3. Inference — 套用到新 DXF

```bash
python gnn-training.py --mode inference \
    --input new_drawing.dxf \
    --model_path best_model.pt \
    --output detected.dxf \
    --conf_threshold 0.7 \
    --dist_threshold 0.02
```

- `conf_threshold` — softmax 信心低於這值會被當作背景丟掉
- `dist_threshold` — 同類別、歸一化距離小於這值的 entity 會被聚成同一個實例
- `--cache_graph some.pt` — 第一次跑會把建好的圖存起來,第二次跑用同一個 DXF
  調 threshold 時可以略過建圖(對大檔很有感)

### 輸出 DXF

`detected.dxf` 是原始 DXF 的副本,**每個元件類別多一個圖層**,命名格式
`DETECTED_<ClassName>`(例如 `DETECTED_Resistor`、`DETECTED_Capacitor`,
名稱中的特殊字元會被替換成底線)。每個 detection 圖層配不同 ACI 顏色,
被偵測到的 entity 會被**複製**到對應圖層(原 entity 不動),用任何
CAD viewer 打開都能直接看到。

---

## 模型概念簡述

每個 DXF entity = 圖上的一個 node,特徵向量 20 維:

- 9 維 entity 類型 one-hot(LINE, ARC, CIRCLE, …)
- 6 維幾何(歸一化的中心、寬高、面積、長寬比)
- 3 維 LINE/ARC 專屬(長度、角度、半徑)
- 1 維 layer id(歸一化)
- 1 維 color

邊用 **同 layer 內 kNN (k=16)** 建,邊權重 = `1/(歐氏距離+ε)`。

模型是 4 層 GATv2 + residual + LayerNorm + MLP head,
邊權重作為 attention bias 餵進去。

---

## 常見問題

**Q: Preprocess 印 "Match rate < 90%" warning 怎麼辦?**
A: JSON 裡的 handle 跟 DXF 對不上,通常是 JSON 來源的 DXF 已經被改過。
重新匯出標註,或檢查 handle 字串大小寫。

**Q: 訓練時 FG Acc 一直是 0?**
A: 多半是類別不平衡 + 收斂前的初期現象。看 epoch 30+ 之後;
若還是 0,檢查 JSON 標註是否真的對到 entity。

**Q: 推論的 DXF 如果有訓練沒見過的 layer?**
A: 該 layer 的 entity 會 fallback 到 layer id 0 並印 `[WARNING]`。
模型沒在這種 layer 上學過,結果會偏掉。建議補充訓練資料。

**Q: 類別數量怎麼設定?**
A: **不需要設定**。Preprocess 從 JSON key 數出有幾類,自動寫進 `cad_graph.pt`,
train/inference 沿著 checkpoint 取用,完全 data-driven。最少 2 類即可運作。
`_write_output_dxf` 的 `COLORS` 表預設 20 色,類別超過 20 時會循環使用顏色,
可在程式裡擴充。

**Q: 不同 dataset 的 `cad_graph.pt` 跟 `best_model.pt` 可以混用嗎?**
A: 不行。`class_names` 跟 `layer2id` 都被綁進 checkpoint;混用會在載入時撞
到 layer 編碼或類別維度不一致。要換 dataset 就重跑整條 pipeline。

---

## 檔名說明

雖然檔名是 `gnn-training.py`,但這支檔同時包含 preprocess / train / inference
三條路徑,並非只做訓練。命名是歷史殘留。
