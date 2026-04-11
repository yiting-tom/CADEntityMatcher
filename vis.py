# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --- 1. 定義輔助計算函數 ---
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def rotate_point(cx, cy, angle_deg, px, py):
    """將點 (px, py) 繞著中心點 (cx, cy) 旋轉"""
    s = math.sin(math.radians(angle_deg))
    c = math.cos(math.radians(angle_deg))
    px -= cx
    py -= cy
    xnew = px * c - py * s
    ynew = px * s + py * c
    return xnew + cx, ynew + cy


# --- 2. 生成模擬資料 ---
print("正在生成模擬資料...")
# 目標 Template (星系指紋)：1大2小，呈 L 型排列
template = [
    {"id": "T1", "type": "ANCHOR", "size": 2.0, "x": 0, "y": 0},  # 太陽 (原點)
    {"id": "T2", "type": "PLANET", "size": 1.0, "x": 5, "y": 0},  # 行星1 (距離 5)
    {"id": "T3", "type": "PLANET", "size": 1.0, "x": 0, "y": 8},  # 行星2 (距離 8)
]

# 生成畫布上的所有點 (包含雜訊與正確的星系)
all_points = []

# 撒下 50 個隨機雜訊
np.random.seed(42)
for _ in range(1000):
    all_points.append(
        {
            "type": np.random.choice(["ANCHOR", "PLANET"]),
            "size": np.random.choice([2.0, 1.0]),
            "x": np.random.uniform(-20, 50),
            "y": np.random.uniform(-20, 50),
        }
    )

# 在畫布中植入 3 個「旋轉且平移」的目標星系
insertions = [
    {"dx": 10, "dy": 10, "angle": 45},  # 傾斜 45 度
    {"dx": 30, "dy": -5, "angle": 120},  # 傾斜 120 度
    {"dx": -5, "dy": 35, "angle": -30},  # 傾斜 -30 度
]

for ins in insertions:
    for t in template:
        # 旋轉
        rx, ry = rotate_point(0, 0, ins["angle"], t["x"], t["y"])
        # 平移
        all_points.append(
            {
                "type": t["type"],
                "size": t["size"],
                "x": rx + ins["dx"],
                "y": ry + ins["dy"],
            }
        )


# --- 3. 核心演算法：星系匹配 ---
print("開始執行全連接星系匹配...")
anchor_template = template[0]
others_template = template[1:]

# 計算 Template 的特徵矩陣
target_features = []
for other in others_template:
    target_features.append(
        {
            "type": other["type"],
            "size": other["size"],
            "dist_to_anchor": calculate_distance(
                (anchor_template["x"], anchor_template["y"]), (other["x"], other["y"])
            ),
        }
    )

# 尋找候選太陽
potential_anchors = [
    p
    for p in all_points
    if p["type"] == anchor_template["type"] and p["size"] == anchor_template["size"]
]

matches_found = []

for cand in potential_anchors:
    cand_pt = (cand["x"], cand["y"])
    matched_neighbors = []

    for target in target_features:
        for p in all_points:
            # 必須不是自己，且類型大小相符
            if (
                p != cand
                and p["type"] == target["type"]
                and p["size"] == target["size"]
            ):
                dist = calculate_distance(cand_pt, (p["x"], p["y"]))
                if abs(dist - target["dist_to_anchor"]) < 0.1:  # 距離容差
                    # 確保不重複加入同一個點
                    if not any(
                        np.allclose((p["x"], p["y"]), (mn["x"], mn["y"]))
                        for mn in matched_neighbors
                    ):
                        matched_neighbors.append(p)
                        break

    # 驗證星系完整性
    if len(matched_neighbors) == len(others_template):
        matches_found.append({"anchor": cand, "neighbors": matched_neighbors})

print(f"掃描完畢！共找到 {len(matches_found)} 個匹配星系。")


# --- 4. Matplotlib 視覺化 ---
print("正在繪製結果...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(
    "Constellation Matching Algorithm (Rigid Body Alignment)",
    fontsize=14,
    fontweight="bold",
)
ax.set_aspect("equal")  # 確保長寬比一致，圓形才不會變橢圓
ax.grid(True, linestyle="--", alpha=0.5)

# 畫出所有散落的點 (背景雜訊)
for p in all_points:
    color = "#CCCCCC" if p["type"] == "PLANET" else "#999999"
    radius = p["size"] * 0.5
    circle = patches.Circle(
        (p["x"], p["y"]), radius, facecolor=color, edgecolor="none", alpha=0.6
    )
    ax.add_patch(circle)

# 標示匹配到的星系
colors = ["#FF3366", "#33CCFF", "#00CC66", "#FF9933"]  # 給不同的匹配群組不同顏色

for i, match in enumerate(matches_found):
    color = colors[i % len(colors)]
    anchor = match["anchor"]
    neighbors = match["neighbors"]

    # 畫太陽 (Anchor) -> 用星星標記
    ax.plot(
        anchor["x"],
        anchor["y"],
        marker="*",
        markersize=15,
        color=color,
        markeredgecolor="black",
    )

    # 計算並畫出這組星系的「群組重心 (Group Centroid)」
    group_x = (anchor["x"] + sum(n["x"] for n in neighbors)) / (1 + len(neighbors))
    group_y = (anchor["y"] + sum(n["y"] for n in neighbors)) / (1 + len(neighbors))
    ax.plot(group_x, group_y, marker="X", markersize=10, color="black")
    ax.text(
        group_x + 1,
        group_y + 1,
        f"Match {i + 1}",
        fontsize=10,
        fontweight="bold",
        color=color,
    )

    # 畫出行星，並畫出連接線 (Rigid Body Links)
    for n in neighbors:
        # 畫行星高亮
        circle = patches.Circle(
            (n["x"], n["y"]),
            n["size"] * 0.6,
            facecolor="none",
            edgecolor=color,
            linewidth=2.5,
        )
        ax.add_patch(circle)
        # 畫連線 (距離矩陣視覺化)
        ax.plot(
            [anchor["x"], n["x"]],
            [anchor["y"], n["y"]],
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
        )

# 圖例設定
ax.plot([], [], marker="o", color="#CCCCCC", ls="", label="Background Noise (Planets)")
ax.plot([], [], marker="o", color="#999999", ls="", label="Background Noise (Anchors)")
ax.plot(
    [],
    [],
    marker="*",
    color="#FF3366",
    ls="",
    markersize=10,
    label="Matched Anchor (Template Base)",
)
ax.plot(
    [],
    [],
    marker="X",
    color="black",
    ls="",
    markersize=8,
    label="Calculated Group Centroid",
)
ax.plot(
    [],
    [],
    color="black",
    linestyle="-",
    linewidth=1.5,
    label="Distance Constraints (Rigid Links)",
)

ax.legend(loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.show()

# %%
