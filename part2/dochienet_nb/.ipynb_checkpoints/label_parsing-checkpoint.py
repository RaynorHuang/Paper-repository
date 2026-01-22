import json
from pathlib import Path

from .data_setup import index_df
def load_label(label_path):
    obj = json.loads(Path(label_path).read_text(encoding="utf-8"))
    pages_meta = obj["pages"]          # dict: page1..pageN -> {width,height}
    contents = obj["contents"]         # list of elements
    
    # 1) sort by reading order
    contents_sorted = sorted(contents, key=lambda x: x.get("order", 10**9))
    
    # 2) collect edges from linking
    edges = set()
    for c in contents_sorted:
        for pair in c.get("linking", []):
            if isinstance(pair, list) and len(pair) == 2:
                edges.add((pair[0], pair[1]))
    return pages_meta, contents_sorted, sorted(list(edges))

# quick test
row = index_df.iloc[0]
pages_meta, elements, edges = load_label(row["label_path"])
print("pages:", len(pages_meta), "elements:", len(elements), "edges:", len(edges))
print("first element keys:", elements[0].keys())
print("first 3 edges:", edges[:3])


# === Notebook cell 3 ===
def build_parent_map(elements, edges, root_id=0):
    parent = {}
    for p, c in edges:
        
        if c in parent and parent[c] != p:
            pass
        else:
            parent[c] = p
    
    
    ids = [e["id"] for e in elements]
    for cid in ids:
        if cid == root_id:
            continue
        if cid not in parent:
            parent[cid] = root_id
    return parent

parent_map = build_parent_map(elements, edges, root_id=0)

for cid in list(parent_map.keys())[:10]:
    print(cid, "->", parent_map[cid])


# === Notebook cell 4 ===
def norm_box_0_1000(box, page_w, page_h):
    x0, y0, x1, y1 = box
    
    nx0 = int(1000 * x0 / page_w)
    nx1 = int(1000 * x1 / page_w)
    ny0 = int(1000 * y0 / page_h)
    ny1 = int(1000 * y1 / page_h)
    # clip
    nx0, ny0 = max(0,nx0), max(0,ny0)
    nx1, ny1 = min(1000,nx1), min(1000,ny1)
    return [nx0, ny0, nx1, ny1]

# test first 5 elements on their pages
def page_size(pages_meta, page_num):
    p = pages_meta[f"page{page_num}"]
    return p["width"], p["height"]

for e in elements[:5]:
    w,h = page_size(pages_meta, e["page"])
    print(e["id"], e["page"], e["label"], norm_box_0_1000(e["box"], w, h))


# === Notebook cell 5 ===
def parse_document(label_path):

    obj = json.loads(Path(label_path).read_text(encoding="utf-8"))
    pages_meta = obj["pages"]
    contents = obj["contents"]

    contents = sorted(contents, key=lambda x: x.get("order", 10**9))

    elements = []
    for c in contents:
        page = c["page"]
        page_info = pages_meta[f"page{page}"]
        pw, ph = page_info["width"], page_info["height"]

        # box → 0–1000（不做 y 翻转）
        x0, y0, x1, y1 = c["box"]
        bbox_1000 = [
            int(1000 * x0 / pw),
            int(1000 * y0 / ph),
            int(1000 * x1 / pw),
            int(1000 * y1 / ph),
        ]

        elements.append({
            "elem_id": c["id"],
            "page_id": page,
            "bbox": bbox_1000,
            "text": c.get("text", ""),
            "label": c.get("label", ""),
            "order": c.get("order", -1),
            "linking": c.get("linking", []),
        })

    return pages_meta, elements


# sanity check
row = index_df.iloc[0]
pages_meta, elements = parse_document(row["label_path"])
print("elements:", len(elements))
for e in elements[:5]:
    print(e)



def build_parent_map(elements, root_id=0):
    """
    elements: list of dict, each must contain:
        - 'id'
        - 'linking'
    return:
        parent_map: dict(child_id -> parent_id), parent_id >= 0
    """
    parent_map = {}

    for e in elements:
        for pair in e.get("linking", []):
            if isinstance(pair, list) and len(pair) == 2:
                p, c = pair
                if c not in parent_map:
                    parent_map[c] = p


    for e in elements:
        cid = e["elem_id"]

        
        if cid not in parent_map:
            parent_map[cid] = root_id

      
        if parent_map[cid] == -1:
            parent_map[cid] = root_id

    return parent_map


# ====== sanity check ======
parent_map = build_parent_map(elements)

bad = [cid for cid, p in parent_map.items() if p < 0]
print("invalid parents (<0):", bad[:10])
print("total elements:", len(elements))
print("total parents:", len(parent_map))


for e in elements[:10]:
    print(f"{e['elem_id']} -> parent {parent_map[e['elem_id']]}")




ROOT_ID = 0


elem_ids = [e["elem_id"] for e in elements]
candidate_parents = [ROOT_ID] + elem_ids


id2idx = {pid: i for i, pid in enumerate(candidate_parents)}


parent_target_idx = []
bad_refs = []

for cid in elem_ids:
    pid = parent_map[cid]          
    if pid not in id2idx:
        bad_refs.append((cid, pid))
        parent_target_idx.append(None)
    else:
        parent_target_idx.append(id2idx[pid])

print("num elements:", len(elem_ids))
print("num candidates (ROOT + elems):", len(candidate_parents))
print("bad parent refs (should be empty):", bad_refs[:10])


for i in range(min(10, len(elem_ids))):
    cid = elem_ids[i]
    pid = parent_map[cid]
    print(f"child {cid:>3} -> parent {pid:>3} | target_idx={parent_target_idx[i]}")


