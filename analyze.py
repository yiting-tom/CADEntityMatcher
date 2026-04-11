# %%
import math
import ezdxf


def calculate_distance(p1, p2):
    """計算兩點之間的直線距離"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def scan_dxf_for_template(dxf_path, template_data):
    """
    使用幾何距離指紋，在 DXF 中尋找符合的 Template 組合。
    支援任意角度旋轉。
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # 1. 整理 Template 資料 (假設從你修改後的 JSON 傳入)
    entities = template_data["entities"]

    # 找出錨點 (這裡用最大半徑的圓當作 Anchor)
    # 你的例子中是 radius 2 的圓
    anchor_template = max(entities, key=lambda e: e["radius"])
    others_template = [e for e in entities if e != anchor_template]

    # 記錄 Template 中小圓與大圓的距離
    anchor_pt = (anchor_template["x"], anchor_template["y"])
    target_distances = []
    for other in others_template:
        pt = (other["x"], other["y"])
        dist = calculate_distance(anchor_pt, pt)
        target_distances.append({"radius": other["radius"], "dist_to_anchor": dist})

    # --- 開始全圖掃描 ---
    matches_found = []

    # 2. 找出全圖所有的圓，並分類
    all_circles = [e for e in msp if e.dxftype() == "CIRCLE"]

    # 3. 找出所有可能是 Anchor 的圓 (半徑為 2)
    # 加入 0.01 的容差避免浮點數誤差
    potential_anchors = [
        c for c in all_circles if abs(c.dxf.radius - anchor_template["radius"]) < 0.01
    ]

    # 4. 驗證每個候選錨點
    for anchor_candidate in potential_anchors:
        cand_pt = (anchor_candidate.dxf.center.x, anchor_candidate.dxf.center.y)

        # 尋找這個錨點周圍的小圓
        matched_neighbors = []
        for target in target_distances:
            # 在所有圓中尋找符合條件的鄰居
            for c in all_circles:
                if abs(c.dxf.radius - target["radius"]) < 0.01:
                    c_pt = (c.dxf.center.x, c.dxf.center.y)
                    dist = calculate_distance(cand_pt, c_pt)

                    # 如果距離與 Template 相符 (容差 0.05)
                    if abs(dist - target["dist_to_anchor"]) < 0.05:
                        # 避免重複加入同一個圓
                        if c not in matched_neighbors:
                            matched_neighbors.append(c)
                            break  # 找到一個符合的就換找下一個 target 特徵

        # 如果找到的鄰居數量與 Template 一致，這就是一個完整的 Match！
        if len(matched_neighbors) == len(others_template):
            matches_found.append(
                {
                    "anchor_x": cand_pt[0],
                    "anchor_y": cand_pt[1],
                    "match_center_x": cand_pt[0],  # 你可以用這個座標來定位你的物件
                    "match_center_y": cand_pt[1],
                }
            )

    return matches_found


# --- 測試使用方式 ---
template_data = {
    "entities": [
        {"type": "CIRCLE", "radius": 0.75, "x": 61.415, "y": 109.82},
        {"type": "CIRCLE", "radius": 0.75, "x": 68.915, "y": 112.07},
        {"type": "CIRCLE", "radius": 2, "x": 64.415, "y": 115.57},
    ]
}

# template_data = { "entities": [
#   {"type": "CIRCLE", "radius": 0.75, "x": 100.1, "y": 110.5},
#   {"type": "CIRCLE", "radius": 0.75, "x": 105.1, "y": 110.5},
#   {"type": "CIRCLE", "radius": 2.0,  "x": 102.6, "y": 115.0}
# ]}
result = scan_dxf_for_template("test.dxf", template_data)
print(f"找到了 {len(result)} 個相同的目標！", result)
# %%

result
