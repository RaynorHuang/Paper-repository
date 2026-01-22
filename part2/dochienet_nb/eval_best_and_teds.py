

# === Notebook cell 19 ===
# CELL EVALBEST: load best checkpoint and evaluate test micro-F1 (all edges / non-root edges)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_model_ssa.pt"
ROOT_ID = 0

# 1) load checkpoint
ckpt = torch.load(CKPT_PATH, map_location=device)
model_ssa.load_state_dict(ckpt["model_state_dict"])
model_ssa.to(device)
model_ssa.eval()

print("Loaded:", CKPT_PATH, "| epoch:", ckpt.get("epoch"), "| best_f1:", ckpt.get("best_f1"))

def micro_prf(model, loader, exclude_root=False, root_id=0):
    tp = pred_n = gt_n = 0
    with torch.no_grad():
        for batch in loader:
            _, logits = model(
                batch["chunks"],
                batch["doc_elem_ids"],
                batch["doc_elem_positions"],
                batch["parent_map"]
            )

            parent_candidates = [root_id] + batch["doc_elem_ids"]
            pred_parent_idx = torch.argmax(logits, dim=1).tolist()

            pred_edges = set((parent_candidates[pidx], child_eid)
                             for child_eid, pidx in zip(batch["doc_elem_ids"], pred_parent_idx))
            gt_edges = set((batch["parent_map"][eid], eid) for eid in batch["doc_elem_ids"])

            if exclude_root:
                pred_edges = set(e for e in pred_edges if e[0] != root_id)
                gt_edges = set(e for e in gt_edges if e[0] != root_id)

            inter = pred_edges & gt_edges
            tp += len(inter)
            pred_n += len(pred_edges)
            gt_n += len(gt_edges)

    p = tp / pred_n if pred_n else 0.0
    r = tp / gt_n if gt_n else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    return p, r, f1, tp, pred_n, gt_n

# 2) evaluate
p, r, f1, tp, pred_n, gt_n = micro_prf(model_ssa, test_loader, exclude_root=False, root_id=ROOT_ID)
print(f"[ALL]    TP={tp} Pred={pred_n} GT={gt_n} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

p2, r2, f12, tp2, pred_n2, gt_n2 = micro_prf(model_ssa, test_loader, exclude_root=True, root_id=ROOT_ID)
print(f"[NONROOT] TP={tp2} Pred={pred_n2} GT={gt_n2} | P={p2:.4f} R={r2:.4f} F1={f12:.4f}")




import sys
import subprocess

try:
    import zss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zss"])
    import zss

ROOT_ID = 0

def build_children_ordered(doc_elem_ids, parent_map, root_id=0):
    order = {eid: i for i, eid in enumerate([root_id] + doc_elem_ids)}
    children = {root_id: []}
    for eid in doc_elem_ids:
        children.setdefault(eid, [])
    for child in doc_elem_ids:
        par = parent_map.get(child, root_id)
        children.setdefault(par, [])
        children[par].append(child)
    for k in children:
        children[k].sort(key=lambda x: order.get(x, 10**9))
    return children

def to_zss_tree(children, node_id, label_map):
    node = zss.Node(label_map.get(node_id, "UNK"))
    for ch in children.get(node_id, []):
        node.addkid(to_zss_tree(children, ch, label_map))
    return node

def predict_parent_map_from_logits(doc_elem_ids, logits, root_id=0):
    parent_candidates = [root_id] + doc_elem_ids
    pred_parent_idx = torch.argmax(logits, dim=1).tolist()
    return {child: parent_candidates[pidx] for child, pidx in zip(doc_elem_ids, pred_parent_idx)}

def compute_semantic_teds_one(doc_elem_ids, gt_parent_map, pred_parent_map, elem_label_map, root_id=0):
    gt_children = build_children_ordered(doc_elem_ids, gt_parent_map, root_id=root_id)
    pr_children = build_children_ordered(doc_elem_ids, pred_parent_map, root_id=root_id)

    # label map: ROOT + element semantic class
    label_map = {root_id: "ROOT"}
    for eid in doc_elem_ids:
        label_map[eid] = elem_label_map.get(eid, "UNK")

    gt_root = to_zss_tree(gt_children, root_id, label_map)
    pr_root = to_zss_tree(pr_children, root_id, label_map)

    def insert_cost(n): return 1
    def remove_cost(n): return 1
    def update_cost(a, b): return 0 if a.label == b.label else 1

    ed = zss.distance(
        gt_root, pr_root,
        get_children=zss.Node.get_children,
        insert_cost=insert_cost,
        remove_cost=remove_cost,
        update_cost=update_cost
    )

    # size: include root + nodes
    denom = 1 + len(doc_elem_ids)
    return 1.0 - (ed / denom)

# ---- Evaluate on test ----
model_ssa.eval()
scores = []

with torch.no_grad():
    for batch in test_loader:
        # IMPORTANT: we need semantic labels per element id.
        # batch currently doesn't include them, so we approximate by re-reading label file via dataset logic:
        # simplest: use the fact that your dataset was built from label json; here we re-open it from index_df.
        # We'll build elem_label_map by scanning the original label json once per doc.

        doc_id = batch["doc_id"]
        # locate label path from index_df
        row = index_df[index_df["doc_id"] == doc_id].iloc[0]
        label_path = row["label_path"]

        obj = json.loads(Path(label_path).read_text(encoding="utf-8"))
        elem_label_map = {c["id"]: c.get("label", "UNK") for c in obj["contents"]}

        _, logits = model_ssa(
            batch["chunks"],
            batch["doc_elem_ids"],
            batch["doc_elem_positions"],
            batch["parent_map"]
        )

        pred_parent_map = predict_parent_map_from_logits(batch["doc_elem_ids"], logits, root_id=ROOT_ID)

        scores.append(
            compute_semantic_teds_one(
                doc_elem_ids=batch["doc_elem_ids"],
                gt_parent_map=batch["parent_map"],
                pred_parent_map=pred_parent_map,
                elem_label_map=elem_label_map,
                root_id=ROOT_ID
            )
        )

avg_teds = sum(scores) / max(len(scores), 1)
print(f"Test docs: {len(scores)}")
print(f"Avg TEDS (semantic-label): {avg_teds:.4f}")

try:
    import zss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zss"])
    import zss

CKPT_PATH = "best_model_ssa_fast.pt"
ROOT_ID = 0

ckpt = torch.load(CKPT_PATH, map_location=device)
model_ssa.load_state_dict(ckpt["model_state_dict"])
model_ssa.to(device)
model_ssa.eval()

print("Loaded:", CKPT_PATH, "| epoch:", ckpt.get("epoch"), "| best_key:", ckpt.get("best_key"))

def micro_prf(model, loader, exclude_root=False, root_id=0):
    model.eval()
    tp = pred_n = gt_n = 0
    with torch.no_grad():
        for batch in loader:
            _, logits = model(
                batch["chunks"],
                batch["doc_elem_ids"],
                batch["doc_elem_positions"],
                batch["parent_map"]
            )
            parent_candidates = [root_id] + batch["doc_elem_ids"]
            pred_parent_idx = torch.argmax(logits, dim=1).tolist()

            pred_edges = set((parent_candidates[pidx], child_eid)
                             for child_eid, pidx in zip(batch["doc_elem_ids"], pred_parent_idx))
            gt_edges = set((batch["parent_map"][eid], eid) for eid in batch["doc_elem_ids"])

            if exclude_root:
                pred_edges = set(e for e in pred_edges if e[0] != root_id)
                gt_edges = set(e for e in gt_edges if e[0] != root_id)

            inter = pred_edges & gt_edges
            tp += len(inter)
            pred_n += len(pred_edges)
            gt_n += len(gt_edges)

    p = tp / pred_n if pred_n else 0.0
    r = tp / gt_n if gt_n else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    return p, r, f1

def build_children_ordered(doc_elem_ids, parent_map, root_id=0):
    order = {eid: i for i, eid in enumerate([root_id] + doc_elem_ids)}
    children = {root_id: []}
    for eid in doc_elem_ids:
        children.setdefault(eid, [])
    for child in doc_elem_ids:
        par = parent_map.get(child, root_id)
        children.setdefault(par, [])
        children[par].append(child)
    for k in children:
        children[k].sort(key=lambda x: order.get(x, 10**9))
    return children

def to_zss_tree(children, node_id, label_map):
    node = zss.Node(label_map.get(node_id, "UNK"))
    for ch in children.get(node_id, []):
        node.addkid(to_zss_tree(children, ch, label_map))
    return node

def semantic_teds_one(doc_id, doc_elem_ids, gt_parent_map, pred_parent_map):
    row = index_df[index_df["doc_id"] == doc_id].iloc[0]
    label_path = row["label_path"]
    obj = json.loads(Path(label_path).read_text(encoding="utf-8"))
    elem_label_map = {c["id"]: c.get("label", "UNK") for c in obj["contents"]}

    gt_children = build_children_ordered(doc_elem_ids, gt_parent_map, root_id=ROOT_ID)
    pr_children = build_children_ordered(doc_elem_ids, pred_parent_map, root_id=ROOT_ID)

    label_map = {ROOT_ID: "ROOT"}
    for eid in doc_elem_ids:
        label_map[eid] = elem_label_map.get(eid, "UNK")

    gt_root = to_zss_tree(gt_children, ROOT_ID, label_map)
    pr_root = to_zss_tree(pr_children, ROOT_ID, label_map)

    def insert_cost(n): return 1
    def remove_cost(n): return 1
    def update_cost(a, b): return 0 if a.label == b.label else 1

    ed = zss.distance(
        gt_root, pr_root,
        get_children=zss.Node.get_children,
        insert_cost=insert_cost,
        remove_cost=remove_cost,
        update_cost=update_cost
    )

    denom = 1 + len(doc_elem_ids)
    return 1.0 - (ed / denom)

def eval_semantic_teds(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            doc_id = batch["doc_id"]
            _, logits = model(
                batch["chunks"],
                batch["doc_elem_ids"],
                batch["doc_elem_positions"],
                batch["parent_map"]
            )
            parent_candidates = [ROOT_ID] + batch["doc_elem_ids"]
            pred_parent_idx = torch.argmax(logits, dim=1).tolist()
            pred_parent_map = {child: parent_candidates[pidx]
                               for child, pidx in zip(batch["doc_elem_ids"], pred_parent_idx)}

            scores.append(
                semantic_teds_one(doc_id, batch["doc_elem_ids"], batch["parent_map"], pred_parent_map)
            )
    return sum(scores)/max(len(scores), 1)
p_all, r_all, f1_all = micro_prf(model_ssa, test_loader, exclude_root=False, root_id=ROOT_ID)
p_nr,  r_nr,  f1_nr  = micro_prf(model_ssa, test_loader, exclude_root=True,  root_id=ROOT_ID)
teds_sem = eval_semantic_teds(model_ssa, test_loader)

print("\n===== FINAL TEST REPORT =====")
print(f"F1_all     : {f1_all:.4f} (P={p_all:.4f}, R={r_all:.4f})")
print(f"F1_nonroot : {f1_nr:.4f} (P={p_nr:.4f}, R={r_nr:.4f})")
print(f"TEDS_sem   : {teds_sem:.4f}")

import math
import json
from pathlib import Path
import types
import sys
import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_ID = 0

# ensure zss
try:
    import zss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zss"])
    import zss

# --- (1) patch model_ssa to support toggles ---
# We override _encode_one_chunk to conditionally add page/inner embeddings.
# This assumes your model_ssa has: encoder, inner_emb, page_proj, page_pe_dim.
def sinusoidal_position_encoding(pos, dim):
    pos = pos.float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=pos.device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(pos.size(0), dim, device=pos.device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# save original method once (if not saved yet)
if not hasattr(model_ssa, "_encode_one_chunk_orig"):
    model_ssa._encode_one_chunk_orig = model_ssa._encode_one_chunk

def _encode_one_chunk_toggled(self, c):
    # base encoder output
    input_ids = torch.tensor(c["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(c["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
    bbox = torch.tensor(c["bbox"], dtype=torch.long, device=device).unsqueeze(0)

    o = self.encoder(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, return_dict=True)
    hidden = o.last_hidden_state.squeeze(0)  # [T,H]

    # toggles
    use_page = getattr(self, "use_page_emb", True)
    use_inner = getattr(self, "use_inner_emb", True)

    if use_page:
        page_ids = torch.tensor(c["page_id"], dtype=torch.long, device=device)
        pe = sinusoidal_position_encoding(page_ids, self.page_pe_dim)  # [T, page_pe_dim]
        page_e = self.page_proj(pe)                                    # [T, H]
        hidden = hidden + page_e

    if use_inner:
        inner_pos = torch.tensor(c["inner_pos"], dtype=torch.long, device=device)
        inner_pos = torch.clamp(inner_pos, 0, self.inner_emb.num_embeddings - 1)
        inner_e = self.inner_emb(inner_pos)                             # [T, H]
        hidden = hidden + inner_e

    return hidden

# bind patched method
model_ssa._encode_one_chunk = types.MethodType(_encode_one_chunk_toggled, model_ssa)

# --- (2) metrics: F1_all / F1_nonroot / TEDS_semantic ---
def micro_f1(model, loader, exclude_root=False, root_id=0):
    model.eval()
    tp = pred_n = gt_n = 0
    with torch.no_grad():
        for batch in loader:
            _, logits = model(
                batch["chunks"],
                batch["doc_elem_ids"],
                batch["doc_elem_positions"],
                batch["parent_map"]
            )
            parent_candidates = [root_id] + batch["doc_elem_ids"]
            pred_parent_idx = torch.argmax(logits, dim=1).tolist()

            pred_edges = set((parent_candidates[pidx], child_eid)
                             for child_eid, pidx in zip(batch["doc_elem_ids"], pred_parent_idx))
            gt_edges = set((batch["parent_map"][eid], eid) for eid in batch["doc_elem_ids"])

            if exclude_root:
                pred_edges = set(e for e in pred_edges if e[0] != root_id)
                gt_edges = set(e for e in gt_edges if e[0] != root_id)

            inter = pred_edges & gt_edges
            tp += len(inter)
            pred_n += len(pred_edges)
            gt_n += len(gt_edges)

    p = tp / pred_n if pred_n else 0.0
    r = tp / gt_n if gt_n else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    model.train()
    return p, r, f1

def build_children_ordered(doc_elem_ids, parent_map, root_id=0):
    order = {eid: i for i, eid in enumerate([root_id] + doc_elem_ids)}
    children = {root_id: []}
    for eid in doc_elem_ids:
        children.setdefault(eid, [])
    for child in doc_elem_ids:
        par = parent_map.get(child, root_id)
        children.setdefault(par, [])
        children[par].append(child)
    for k in children:
        children[k].sort(key=lambda x: order.get(x, 10**9))
    return children

def to_zss_tree(children, node_id, label_map):
    node = zss.Node(label_map.get(node_id, "UNK"))
    for ch in children.get(node_id, []):
        node.addkid(to_zss_tree(children, ch, label_map))
    return node

def semantic_teds_one(doc_id, doc_elem_ids, gt_parent_map, pred_parent_map):
    row = index_df[index_df["doc_id"] == doc_id].iloc[0]
    label_path = row["label_path"]
    obj = json.loads(Path(label_path).read_text(encoding="utf-8"))
    elem_label_map = {c["id"]: c.get("label", "UNK") for c in obj["contents"]}

    gt_children = build_children_ordered(doc_elem_ids, gt_parent_map, root_id=ROOT_ID)
    pr_children = build_children_ordered(doc_elem_ids, pred_parent_map, root_id=ROOT_ID)

    label_map = {ROOT_ID: "ROOT"}
    for eid in doc_elem_ids:
        label_map[eid] = elem_label_map.get(eid, "UNK")

    gt_root = to_zss_tree(gt_children, ROOT_ID, label_map)
    pr_root = to_zss_tree(pr_children, ROOT_ID, label_map)

    def insert_cost(n): return 1
    def remove_cost(n): return 1
    def update_cost(a, b): return 0 if a.label == b.label else 1

    ed = zss.distance(
        gt_root, pr_root,
        get_children=zss.Node.get_children,
        insert_cost=insert_cost,
        remove_cost=remove_cost,
        update_cost=update_cost
    )

    denom = 1 + len(doc_elem_ids)
    return 1.0 - (ed / denom)

def eval_semantic_teds(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            doc_id = batch["doc_id"]
            _, logits = model(
                batch["chunks"],
                batch["doc_elem_ids"],
                batch["doc_elem_positions"],
                batch["parent_map"]
            )
            parent_candidates = [ROOT_ID] + batch["doc_elem_ids"]
            pred_parent_idx = torch.argmax(logits, dim=1).tolist()
            pred_parent_map = {child: parent_candidates[pidx]
                               for child, pidx in zip(batch["doc_elem_ids"], pred_parent_idx)}
            scores.append(semantic_teds_one(doc_id, batch["doc_elem_ids"], batch["parent_map"], pred_parent_map))
    model.train()
    return sum(scores) / max(len(scores), 1)

def run_one(name, use_page, use_inner):
    model_ssa.use_page_emb = use_page
    model_ssa.use_inner_emb = use_inner

    p_all, r_all, f1_all = micro_f1(model_ssa, test_loader, exclude_root=False, root_id=ROOT_ID)
    p_nr,  r_nr,  f1_nr  = micro_f1(model_ssa, test_loader, exclude_root=True,  root_id=ROOT_ID)
    teds = eval_semantic_teds(model_ssa, test_loader)

    print(f"\n=== {name} ===")
    print(f"use_page_emb={use_page} | use_inner_emb={use_inner}")
    print(f"F1_all     : {f1_all:.4f} (P={p_all:.4f}, R={r_all:.4f})")
    print(f"F1_nonroot : {f1_nr:.4f} (P={p_nr:.4f}, R={r_nr:.4f})")
    print(f"TEDS_sem   : {teds:.4f}")
    return {"name": name, "F1_all": f1_all, "F1_nonroot": f1_nr, "TEDS_sem": teds}

# --- (3) run ablations ---
model_ssa.to(device)

res = []
res.append(run_one("baseline (PageE+InnerE)", True, True))
res.append(run_one("w/o PageE", False, True))
res.append(run_one("w/o InnerE", True, False))

print("\n===== SUMMARY (copy to report) =====")
for r in res:
    print(f'{r["name"]}: F1_all={r["F1_all"]:.4f} | F1_nonroot={r["F1_nonroot"]:.4f} | TEDS_sem={r["TEDS_sem"]:.4f}')



