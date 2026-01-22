import os
import glob
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset

ID2LABEL_14 = [
    "Title", "Author", "Mail", "Affiliation", "Section",
    "First-Line", "Para-Line", "Equation", "Table", "Figure",
    "Caption", "Page-Footer", "Page-Header", "Footnote"
]
LABEL2ID_14 = {k:i for i,k in enumerate(ID2LABEL_14)}

REL2ID = {"connect": 0, "contain": 1, "equality": 2, "meta": 3}
ID2REL = {v:k for k,v in REL2ID.items()}


def map_hrd_class_to_14(c: str) -> int:
    c = (c or "").lower().strip()
    if c == "title":
        return LABEL2ID_14["Title"]
    if c == "author":
        return LABEL2ID_14["Author"]
    if c in ("affili", "affiliation"):
        return LABEL2ID_14["Affiliation"]
    if c in ("header",):
        return LABEL2ID_14["Page-Header"]
    if c in ("footer",):
        return LABEL2ID_14["Page-Footer"]
    if c in ("fnote",):
        return LABEL2ID_14["Footnote"]
    if c.startswith("sec"):
        return LABEL2ID_14["Section"]
    if c in ("fstline",):
        return LABEL2ID_14["First-Line"]
    if c in ("para", "opara"):
        return LABEL2ID_14["Para-Line"]
    # 兜底
    return LABEL2ID_14["Para-Line"]



def get_image_path(image_root, doc_id, page_id):
    """
    Try multiple naming conventions:
    1) HRDH-style: <image_root>/<doc_id>/<page_id>.png
    2) HRDH-style with other ext: <image_root>/<doc_id>/<page_id>.(jpg/jpeg/png)
    3) HRDS-style: <image_root>/<doc_id>/<doc_id>_<page_id>.(jpg/jpeg/png)
    4) HRDS-style flat (just in image_root): <image_root>/<doc_id>_<page_id>.(jpg/jpeg/png)

    Returns existing path; otherwise raises FileNotFoundError with helpful info.
    """
    exts = ("png", "jpg", "jpeg", "webp")

    # --- A) in subfolder <doc_id>/ ---
    doc_dir = os.path.join(image_root, doc_id)

    # 1) <page_id>.<ext>
    for ext in exts:
        p = os.path.join(doc_dir, f"{page_id}.{ext}")
        if os.path.exists(p):
            return p

    # 2) <doc_id>_<page_id>.<ext>
    for ext in exts:
        p = os.path.join(doc_dir, f"{doc_id}_{page_id}.{ext}")
        if os.path.exists(p):
            return p

    # --- B) flat in image_root ---
    for ext in exts:
        p = os.path.join(image_root, f"{doc_id}_{page_id}.{ext}")
        if os.path.exists(p):
            return p

    # --- C) glob fallback (covers weird ext/case) ---
    # in doc folder
    pats = [
        os.path.join(doc_dir, f"{page_id}.*"),
        os.path.join(doc_dir, f"{doc_id}_{page_id}.*"),
        os.path.join(image_root, f"{doc_id}_{page_id}.*"),
    ]
    for pat in pats:
        hits = glob.glob(pat)
        if hits:
            # pick the first deterministically (sorted)
            hits = sorted(hits)
            return hits[0]

    raise FileNotFoundError(
        f"Missing page image for doc_id={doc_id}, page_id={page_id}. "
        f"Tried under: {doc_dir} and {image_root}. "
        f"Example expected: {os.path.join(doc_dir, str(page_id)+'.png')} or {os.path.join(doc_dir, f'{doc_id}_{page_id}.jpg')}"
    )

import heapq
from collections import defaultdict

def topo_sort_with_reading_priority(items):
    """
    items: HRDH json list，每个元素至少包含 box/page/parent_id
    返回：满足 parent 一定在 child 前的顺序（同时尽量贴近阅读序）
    """
    n = len(items)

    # reading priority rank
    ranks = []
    for i, it in enumerate(items):
        x0, y0, x1, y1 = it["box"]
        ranks.append((int(it["page"]), float(y0), float(x0), i))

    # parent -> child graph
    indeg = [0] * n
    g = defaultdict(list)
    for child in range(n):
        p = int(items[child].get("parent_id", -1))
        if p < 0:
            continue
        if 0 <= p < n:
            g[p].append(child)
            indeg[child] += 1

    # Kahn with heap prioritized by reading rank
    heap = []
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(heap, (ranks[i], i))

    order = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (ranks[v], v))

    # fallback if cycles/noise exist
    if len(order) < n:
        remaining = [i for i in range(n) if i not in set(order)]
        remaining.sort(key=lambda i: ranks[i])
        order.extend(remaining)

    return order


def load_hrdh_json(json_path: str) -> Dict[str, Any]:
    """
    读取单个 HRDH json，返回 doc dict（units + labels + parent + relation），并重映射 parent 到新排序 index。
    当前排序：按 (page, y0, x0) 近似阅读顺序（足够用于训练闭环；后续可替换为双栏阅读序）。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    def sort_key(idx_item):
        idx, it = idx_item
        x0, y0, x1, y1 = it["box"]
        return (int(it["page"]), int(y0), int(x0), idx)

    order_old = topo_sort_with_reading_priority(items)
    indexed_sorted = [(i, items[i]) for i in order_old]


    old2new = {old_i: new_i for new_i, (old_i, _) in enumerate(indexed_sorted)}

    units = []
    y_parent, y_rel, y_cls, is_meta = [], [], [], []

    for new_i, (old_i, it) in enumerate(indexed_sorted):
        text = (it.get("text") or "").strip()
        x0, y0, x1, y1 = it["box"]
        page_id = int(it["page"])
        cls_raw = it.get("class", "para")
        rel_raw = it.get("relation", "connect")
        meta_flag = bool(it.get("is_meta", False))
        parent_old = int(it.get("parent_id", -1))
        parent_new = -1 if parent_old == -1 else old2new.get(parent_old, -1)

        units.append({
            "text": text,
            "bbox": (float(x0), float(y0), float(x1), float(y1)),  # pixel bbox
            "page_id": page_id,
            "order_id": new_i,
            "class_raw": cls_raw,
        })
        y_parent.append(parent_new)
        y_rel.append(REL2ID.get(rel_raw, 0))
        y_cls.append(map_hrd_class_to_14(cls_raw))
        is_meta.append(meta_flag)

    doc_id = os.path.basename(json_path).replace(".json", "")
    return {
        "doc_id": doc_id,
        "json_path": json_path,
        "units": units,
        "y_parent": y_parent,
        "y_rel": y_rel,
        "y_cls": y_cls,
        "is_meta": is_meta,
    }


class HRDHDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", max_len: int = 512,cfg=None):
        """
        root_dir: .../HRDH
        split: train 或 test
        max_len: 截断长度（论文常用 512）
        """
        assert split in ("train", "test")
        self.root_dir = root_dir
        self.split = split
        self.max_len = max_len

        self.json_dir = os.path.join(root_dir, split)
        self.image_root = os.path.join(root_dir, "images")  

        self.json_paths = sorted(glob.glob(os.path.join(self.json_dir, "*.json")))
        if not self.json_paths:
            raise FileNotFoundError(f"No json files found in: {self.json_dir}")

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        json_path = self.json_paths[idx]
        doc = load_hrdh_json(json_path)

        # 截断（保持 parent 合法）
        if len(doc["units"]) > self.max_len:
            keep = self.max_len
            doc["units"] = doc["units"][:keep]
            doc["y_cls"] = doc["y_cls"][:keep]
            doc["y_rel"] = doc["y_rel"][:keep]
            doc["is_meta"] = doc["is_meta"][:keep]
            # parent: 超出范围的 parent 置为 0；指向被截断的也置 0（或 -1）
            y_parent = []
            for i, p in enumerate(doc["y_parent"][:keep]):
                if p == -1:
                    y_parent.append(-1)
                elif 0 <= p < keep:
                    y_parent.append(p)
                else:
                    y_parent.append(-1)
            doc["y_parent"] = y_parent

        # 为每个 unit 找到对应页图路径（按 page_id）
        # 注意：同一页只加载一次图片，训练时可以再做缓存/预处理
        page_ids = sorted(set(u["page_id"] for u in doc["units"]))
        page_images = {}
        for pid in page_ids:
            page_images[pid] = get_image_path(self.image_root, doc["doc_id"], pid)
        doc["page_images"] = page_images
        for i, p in enumerate(doc["y_parent"]):
            if p >= 0:
                assert p < i, f"parent not causal: i={i}, p={p}"
            if "is_meta" in doc:
                for i, flag in enumerate(doc["is_meta"]):
                    if flag:
                        doc["y_parent"][i] = -1  # ROOT
        
        return doc


REL3 = {"connect": 0, "contain": 1, "equality": 2}  # 论文三类
REL3_INV = {v:k for k,v in REL3.items()}

def filter_meta_and_remap(units_raw):
    """
    units_raw: list of dict from json, each has:
      text, box, class, page, is_meta, parent_id, relation
    return:
      units_kept: list of dict with keys {text, box, page_id, cls_name, parent, rel}
      old2new: dict old_index -> new_index  (only for kept)
    """
    keep_idx = [i for i,u in enumerate(units_raw) if not u.get("is_meta", False)]
    old2new = {old:new for new,old in enumerate(keep_idx)}

    units_kept = []
    for old_i in keep_idx:
        u = units_raw[old_i]
        p_old = int(u.get("parent_id", -1))
        # 如果 parent 被过滤掉，或本来就是 -1，则设为 ROOT(-1)
        p_new = old2new.get(p_old, -1) if p_old >= 0 else -1

        rel = u.get("relation", None)
        # 过滤后不应该再出现 meta；如果仍然出现，直接跳过该样本或置默认
        if rel == "meta":
            # 这里选择：直接将该单元丢弃（更干净）
            continue

        if rel not in REL3:
            raise ValueError(f"Unknown relation: {rel}")

        units_kept.append({
            "text": u.get("text",""),
            "box": u.get("box"),
            "page_id": int(u.get("page", 0)),
            "cls_name": u.get("class"),
            "parent": p_new,
            "rel": REL3[rel],
        })

    return units_kept, old2new
