from __future__ import annotations

from .metrics import compute_steds, load_pair_files


import json, os, re, numpy as np
from tqdm import tqdm

# ---------- text normalization ----------
def _norm_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("- ", "-")
    return s


# ---------- STEDS variants ----------
def steds_original(gt_js, pr_js):
    return compute_steds(gt_js, pr_js, exclude_meta=True)


def steds_label_only(gt_js, pr_js):
    gt_root, gt_n = build_tree_from_export(gt_js, exclude_meta=True)
    pr_root, pr_n = build_tree_from_export(pr_js, exclude_meta=True)

    def wipe_text(x):
        x.text = ""
        for c in x.children:
            wipe_text(c)

    wipe_text(gt_root)
    wipe_text(pr_root)

    dist = ted_zhang_shasha(gt_root, pr_root)
    denom = max(gt_n, pr_n, 1)
    return 1.0 - dist / denom


def steds_text_norm(gt_js, pr_js):
    gt_root, gt_n = build_tree_from_export(gt_js, exclude_meta=True)
    pr_root, pr_n = build_tree_from_export(pr_js, exclude_meta=True)

    def apply_norm(x):
        x.text = _norm_text(x.text)
        for c in x.children:
            apply_norm(c)

    apply_norm(gt_root)
    apply_norm(pr_root)

    dist = ted_zhang_shasha(gt_root, pr_root)
    denom = max(gt_n, pr_n, 1)
    return 1.0 - dist / denom


# ---------- batch evaluation ----------
def eval_steds_variants(export_dir: str):
    pairs = load_pair_files(export_dir)

    res = {
        "original": [],
        "label_only": [],
        "text_norm": [],
    }

    for gt_path, pr_path in tqdm(pairs, desc="STEDS variants"):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        with open(pr_path, "r", encoding="utf-8") as f:
            pr = json.load(f)

        res["original"].append(steds_original(gt, pr))
        res["label_only"].append(steds_label_only(gt, pr))
        res["text_norm"].append(steds_text_norm(gt, pr))

    def summary(arr):
        arr = np.asarray(arr)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    report = {k: summary(v) for k, v in res.items()}
    return report


# ---------- run ----------



import numpy as np

def kmeans_1d(x, k, iters=30):
    """简单 1D kmeans。x: (n,) float"""
    x = x.astype(np.float32)
    # init: 均匀取 k 个分位点
    qs = np.linspace(0.1, 0.9, k)
    centers = np.quantile(x, qs)
    for _ in range(iters):
        # assign
        d = np.abs(x[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = x[mask].mean()
        if np.allclose(new_centers, centers, atol=1e-4):
            break
        centers = new_centers
    # inertia
    inertia = ((x - centers[labels]) ** 2).sum()
    # sort clusters by center (left->right)
    order = np.argsort(centers)
    remap = np.zeros_like(order)
    remap[order] = np.arange(k)
    labels = remap[labels]
    centers = centers[order]
    return labels, centers, inertia

def choose_k_1d(x, k_min=1, k_max=3, min_cluster_size=5):
    """
    自动选择列数 k：
    - 基于“惯性下降幅度”（elbow-ish）
    - 同时避免产生很小的列
    """
    best = None
    prev_inertia = None
    for k in range(k_min, k_max + 1):
        labels, centers, inertia = kmeans_1d(x, k)
        # 小簇过滤
        ok = True
        for j in range(k):
            if (labels == j).sum() < min_cluster_size and len(x) >= min_cluster_size * k:
                ok = False
                break
        if not ok:
            continue

        if prev_inertia is None:
            score = 0.0
        else:
            # inertia 下降比例越大越好
            score = (prev_inertia - inertia) / max(prev_inertia, 1e-6)

        # 记录：优先更大“下降”，其次更小 inertia
        cand = (score, -inertia, k, labels, centers, inertia)
        if best is None or cand > best:
            best = cand
        prev_inertia = inertia

    if best is None:
        labels, centers, inertia = kmeans_1d(x, 1)
        return 1, labels, centers
    _, _, k, labels, centers, _ = best
    return k, labels, centers


def order_units_multi_column(units, page_w, page_h, max_cols=3):
    """
    units: list of dict with 'box'=[x0,y0,x1,y1]
    返回：按阅读顺序排序后的 index 列表
    """
    n = len(units)
    if n <= 1:
        return list(range(n))

    boxes = np.array([u["box"] for u in units], dtype=np.float32)
    x0, y0, x1, y1 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    xc = 0.5 * (x0 + x1)
    yt = y0
    bw = np.clip(x1 - x0, 1.0, None)

    # 识别“跨栏大框”：宽度超过页面的一定比例（标题/大图/大表等），单独处理
    wide = bw > (0.70 * page_w)

    # 对非 wide 的做列聚类
    idx_normal = np.where(~wide)[0]
    idx_wide = np.where(wide)[0]

    ordered = []

    # 先把“宽大框”按 y 排到合适位置：通常它们是跨栏标题/大图，阅读上仍按 y 走
    # 策略：把 wide 与 normal 一起排序时，wide 作为“单独列”插入：这里用更稳的两阶段合并
    if len(idx_normal) > 0:
        x_norm = xc[idx_normal]
        k, labels, centers = choose_k_1d(x_norm, 1, max_cols, min_cluster_size=4)

        # 每一列内部按 y_top 排序
        cols = []
        for c in range(k):
            members = idx_normal[labels == c]
            members = members[np.argsort(yt[members])]
            cols.append(list(members))

        # 列从左到右拼接
        seq_normal = [i for col in cols for i in col]
    else:
        seq_normal = []

    # wide 按 y_top 排序
    seq_wide = list(idx_wide[np.argsort(yt[idx_wide])]) if len(idx_wide) > 0 else []

    # 合并 wide 与 normal：按 y_top 进行稳定插入（wide 的 y 决定位置）
    # 做法：对两序列按 y_top 归并，但当 y 接近时优先 wide（跨栏标题通常应先读）
    def merge_by_y(a, b):
        ia = ib = 0
        out = []
        while ia < len(a) and ib < len(b):
            ya = yt[a[ia]]
            yb = yt[b[ib]]
            if yb <= ya + 2:  # 容忍 2px 抖动，wide 略优先
                out.append(b[ib]); ib += 1
            else:
                out.append(a[ia]); ia += 1
        out.extend(a[ia:])
        out.extend(b[ib:])
        return out

    ordered = merge_by_y(seq_normal, seq_wide)
    return ordered

def sort_doc_units_by_reading_order(doc_units, page_images):
    """
    doc_units: list[dict], each has page_id, box, text, ...
    page_images: dict {page_id: image_path}
    return sorted_units
    """
    # 1) 按页收集
    by_page = {}
    for u in doc_units:
        pid = int(u["page_id"])
        by_page.setdefault(pid, []).append(u)

    # 2) 页按 pid 排序
    sorted_units = []
    for pid in sorted(by_page.keys()):
        # 读 page 尺寸
        from PIL import Image
        img = Image.open(page_images[pid])
        W, H = img.width, img.height

        units_p = by_page[pid]
        order_idx = order_units_multi_column(units_p, W, H, max_cols=3)
        sorted_units.extend([units_p[i] for i in order_idx])

    return sorted_units

import numpy as np
from PIL import Image

def _kmeans_1d(x, k, iters=30):
    x = x.astype(np.float32)
    qs = np.linspace(0.1, 0.9, k)
    centers = np.quantile(x, qs)

    for _ in range(iters):
        d = np.abs(x[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                new_centers[j] = x[m].mean()
        if np.allclose(new_centers, centers, atol=1e-4):
            break
        centers = new_centers

    inertia = ((x - centers[labels]) ** 2).sum()

    order = np.argsort(centers)
    remap = np.zeros_like(order)
    remap[order] = np.arange(k)
    labels = remap[labels]
    centers = centers[order]
    return labels, centers, inertia


def _choose_k_1d(x, k_min=1, k_max=3, min_cluster_size=4):
    best = None
    prev_inertia = None

    for k in range(k_min, k_max + 1):
        labels, centers, inertia = _kmeans_1d(x, k)
        ok = True
        if len(x) >= min_cluster_size * k:
            for j in range(k):
                if (labels == j).sum() < min_cluster_size:
                    ok = False
                    break
        if not ok:
            continue

        score = 0.0 if prev_inertia is None else (prev_inertia - inertia) / max(prev_inertia, 1e-6)
        cand = (score, -inertia, k, labels, centers)
        if best is None or cand > best:
            best = cand
        prev_inertia = inertia

    if best is None:
        labels, centers, _ = _kmeans_1d(x, 1)
        return 1, labels, centers
    _, _, k, labels, centers = best
    return k, labels, centers


def order_units_multi_column(units, page_w, page_h, max_cols=3):
    """
    units: list of dict with 'box'=[x0,y0,x1,y1]
    return: list of indices in reading order
    """
    n = len(units)
    if n <= 1:
        return list(range(n))

    boxes = np.array([u["box"] for u in units], dtype=np.float32)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xc = 0.5 * (x0 + x1)
    yt = y0
    bw = np.clip(x1 - x0, 1.0, None)

    # “跨栏大框”识别：标题/大图/大表常见
    wide = bw > (0.70 * page_w)

    idx_normal = np.where(~wide)[0]
    idx_wide = np.where(wide)[0]

    seq_normal = []
    if len(idx_normal) > 0:
        x_norm = xc[idx_normal]
        k, labels, _ = _choose_k_1d(x_norm, 1, max_cols, min_cluster_size=4)

        cols = []
        for c in range(k):
            members = idx_normal[labels == c]
            members = members[np.argsort(yt[members])]
            cols.append(list(members))

        seq_normal = [i for col in cols for i in col]

    seq_wide = list(idx_wide[np.argsort(yt[idx_wide])]) if len(idx_wide) > 0 else []

    # 按 y 归并：wide 在接近同一 y 时略优先（跨栏标题通常应先读）
    def merge_by_y(a, b):
        ia = ib = 0
        out = []
        while ia < len(a) and ib < len(b):
            ya = yt[a[ia]]
            yb = yt[b[ib]]
            if yb <= ya + 2:
                out.append(b[ib]); ib += 1
            else:
                out.append(a[ia]); ia += 1
        out.extend(a[ia:])
        out.extend(b[ib:])
        return out

    return merge_by_y(seq_normal, seq_wide)


def sort_doc_units_by_reading_order(doc_units, page_images, max_cols=3):
    """
    doc_units: list[dict] each has 'page_id','box',...
    page_images: dict[int -> image_path]
    return: new_units_sorted
    """
    by_page = {}
    for u in doc_units:
        pid = int(u["page_id"])
        by_page.setdefault(pid, []).append(u)

    sorted_units = []
    for pid in sorted(by_page.keys()):
        img = Image.open(page_images[pid])
        W, H = img.width, img.height

        units_p = by_page[pid]
        order_idx = order_units_multi_column(units_p, W, H, max_cols=max_cols)
        sorted_units.extend([units_p[i] for i in order_idx])

    return sorted_units
