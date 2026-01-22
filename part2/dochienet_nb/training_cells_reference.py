
import time
import math
import random
import numpy as np
import torch
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "现在不是 CUDA。"

# reproducibility (optional)
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

EPOCHS = 5               
LR_START = 4e-5
LR_END = 1e-6
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
PRINT_EVERY_STEP = 20
SAVE_PATH = "best_model_ssa.pt"
best_f1 = -1.0

model_ssa.to(device)
model_ssa.train()

optimizer = optim.AdamW(model_ssa.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)
gamma = (LR_END / LR_START) ** (1.0 / max(EPOCHS - 1, 1))
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

scaler = torch.cuda.amp.GradScaler(enabled=True)

def micro_f1_on_loader(model, loader, root_id=0):
    model.eval()
    tp_sum = pred_sum = gt_sum = 0
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss, logits = model(
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
            inter = gt_edges & pred_edges
            tp_sum += len(inter); pred_sum += len(pred_edges); gt_sum += len(gt_edges)
    p = tp_sum / pred_sum if pred_sum else 0.0
    r = tp_sum / gt_sum if gt_sum else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    model.train()
    return p, r, f1

global_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    step_times = []

    total_loss = 0.0
    steps = 0

    for step, batch in enumerate(train_loader, start=1):
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss, _ = model_ssa(
                batch["chunks"],
                batch["doc_elem_ids"],
                batch["doc_elem_positions"],
                batch["parent_map"]
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_ssa.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        steps += 1

        dt = time.time() - t0
        step_times.append(dt)

        if step % PRINT_EVERY_STEP == 0:
            avg_dt = sum(step_times[-PRINT_EVERY_STEP:]) / PRINT_EVERY_STEP
            print(f"[epoch {epoch:03d} | step {step:04d}] loss={loss.item():.4f} | {avg_dt:.2f}s/step")

    avg_train_loss = total_loss / max(steps, 1)

    eval_start = time.time()
    p, r, f1 = micro_f1_on_loader(model_ssa, test_loader, root_id=0)
    eval_time = time.time() - eval_start

    # save best
    if f1 > best_f1:
        best_f1 = f1
        torch.save({"epoch": epoch, "model_state_dict": model_ssa.state_dict(), "best_f1": best_f1}, SAVE_PATH)
        tag = " (best)"
    else:
        tag = ""

    epoch_time = time.time() - epoch_start
    elapsed = time.time() - global_start
    avg_epoch = elapsed / epoch
    eta = avg_epoch * (EPOCHS - epoch)
    lr = optimizer.param_groups[0]["lr"]

    print(
        f"\n[epoch {epoch:03d}/{EPOCHS}] lr={lr:.2e} | train_loss={avg_train_loss:.4f} | test_F1={f1:.4f}{tag}\n"
        f"  epoch_time={epoch_time/60:.1f} min | eval_time={eval_time:.1f}s | ETA={eta/60:.1f} min\n"
    )

    scheduler.step()

print("done. best_f1=", best_f1, "saved:", SAVE_PATH)
'''

# === Notebook cell 22 ===
# TRAIN CELL (FINAL, REPLACE): AMP + grad accumulation + warmup/decay + step/epoch timing + ETA
# + train-step cap per epoch + sparse eval (TEDS less frequent) + optional chunk-cap patch
#
# Assumes you already have: model_ssa, DocHieNetDocDataset, index_df, collate_fn
# This cell rebuilds train_loader/test_loader with faster caps, then trains.
'''
import time
import math
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    import zss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zss"])
    import zss

# ------------------ reproducibility ------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ------------------ speed knobs (RTX 4060 8GB) ------------------
EPOCHS = 10                       
MAX_TRAIN_STEPS_PER_EPOCH = 200   
PRINT_EVERY_STEP = 20

TRAIN_MAX_CHUNKS = 16             
TEST_MAX_CHUNKS  = 64             

EVAL_EVERY_EPOCH = 1              
EVAL_TEDS_EVERY  = 5              

# ------------------ optimization ------------------
LR_START = 4e-5
LR_END = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
GRAD_ACCUM_STEPS = 2              
MAX_GRAD_NORM = 1.0
ROOT_ID = 0

SAVE_PATH = "best_model_ssa_fast.pt"
# choose best by: "F1_nonroot" or "TEDS_sem"
BEST_BY = "F1_nonroot"

# ------------------ rebuild loaders with new caps ------------------
train_ds = DocHieNetDocDataset(index_df, split="train")
test_ds  = DocHieNetDocDataset(index_df, split="test")
train_ds.max_chunks = TRAIN_MAX_CHUNKS
test_ds.max_chunks  = TEST_MAX_CHUNKS

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

print("Loader caps:", f"train_max_chunks={TRAIN_MAX_CHUNKS}", f"test_max_chunks={TEST_MAX_CHUNKS}")
print("Train steps/epoch cap:", MAX_TRAIN_STEPS_PER_EPOCH)

# ------------------ model/optimizer ------------------
model_ssa.to(device)
model_ssa.train()

optimizer = optim.AdamW(model_ssa.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=True)

# total updates = epochs * ceil(train_steps/accum)
steps_per_epoch = math.ceil(MAX_TRAIN_STEPS_PER_EPOCH / GRAD_ACCUM_STEPS)
total_updates = steps_per_epoch * EPOCHS
warmup_updates = max(1, int(total_updates * WARMUP_RATIO))

def set_lr(lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

def lr_schedule(update_idx):
    if update_idx <= warmup_updates:
        return LR_START * (update_idx / warmup_updates)
    t = update_idx - warmup_updates
    T = max(1, total_updates - warmup_updates)
    gamma = (LR_END / LR_START) ** (1.0 / T)
    return LR_START * (gamma ** t)

# ------------------ eval helpers ------------------
def micro_f1(model, loader, exclude_root=False, root_id=0):
    model.eval()
    tp = pred_n = gt_n = 0
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(dtype=torch.float16):
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
            tp += len(inter); pred_n += len(pred_edges); gt_n += len(gt_edges)

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
            with torch.cuda.amp.autocast(dtype=torch.float16):
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
            scores.append(semantic_teds_one(batch["doc_id"], batch["doc_elem_ids"], batch["parent_map"], pred_parent_map))
    model.train()
    return sum(scores) / max(len(scores), 1)

# ------------------ train loop ------------------
global_start = time.time()
global_update = 0

best_key = -1.0
best_snapshot = None

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    step_times = []

    model_ssa.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    steps = 0
    updates = 0

    for step, batch in enumerate(train_loader, start=1):
        t0 = time.time()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss, _ = model_ssa(
                batch["chunks"],
                batch["doc_elem_ids"],
                batch["doc_elem_positions"],
                batch["parent_map"]
            )

        scaler.scale(loss / GRAD_ACCUM_STEPS).backward()

        total_loss += float(loss.detach().cpu())
        steps += 1

        dt = time.time() - t0
        step_times.append(dt)

        if step % PRINT_EVERY_STEP == 0:
            avg_dt = sum(step_times[-PRINT_EVERY_STEP:]) / PRINT_EVERY_STEP
            print(f"[epoch {epoch:03d} | step {step:04d}] loss={loss.item():.4f} | {avg_dt:.2f}s/step")

        # optimizer update
        if step % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_ssa.parameters(), MAX_GRAD_NORM)

            global_update += 1
            lr = lr_schedule(global_update)
            set_lr(lr)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

        if step >= MAX_TRAIN_STEPS_PER_EPOCH:
            break

    avg_train_loss = total_loss / max(steps, 1)

    # ------------------ eval ------------------
    eval_start = time.time()

    if (epoch % EVAL_EVERY_EPOCH) == 0:
        _, _, f1_all = micro_f1(model_ssa, test_loader, exclude_root=False, root_id=ROOT_ID)
        _, _, f1_nr  = micro_f1(model_ssa, test_loader, exclude_root=True,  root_id=ROOT_ID)
    else:
        f1_all, f1_nr = float("nan"), float("nan")

    teds_sem = float("nan")
    if (epoch % EVAL_TEDS_EVERY) == 0:
        teds_sem = eval_semantic_teds(model_ssa, test_loader)

    eval_time = time.time() - eval_start

    # ------------------ choose best & save ------------------
    if BEST_BY == "F1_nonroot":
        key = f1_nr
    else:
        key = teds_sem

    tag = ""
    if not (isinstance(key, float) and (math.isnan(key))):
        if key > best_key:
            best_key = key
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_ssa.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_key": best_key,
                    "best_by": BEST_BY,
                    "f1_all": f1_all,
                    "f1_nonroot": f1_nr,
                    "teds_sem": teds_sem,
                    "seed": SEED,
                    "global_update": global_update,
                    "caps": {"train_max_chunks": TRAIN_MAX_CHUNKS, "test_max_chunks": TEST_MAX_CHUNKS},
                    "train_steps_per_epoch_cap": MAX_TRAIN_STEPS_PER_EPOCH,
                    "grad_accum_steps": GRAD_ACCUM_STEPS,
                    "note": "fast timed train for RTX 4060 8GB"
                },
                SAVE_PATH
            )
            tag = " (best)"

    # ------------------ timing / ETA ------------------
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - global_start
    avg_epoch = elapsed / epoch
    eta = avg_epoch * (EPOCHS - epoch)
    lr_now = optimizer.param_groups[0]["lr"]

    # pretty print
    f1_all_str  = f"{f1_all:.4f}" if not (isinstance(f1_all, float) and math.isnan(f1_all)) else "skip"
    f1_nr_str   = f"{f1_nr:.4f}"  if not (isinstance(f1_nr, float) and math.isnan(f1_nr))  else "skip"
    teds_str    = f"{teds_sem:.4f}" if not (isinstance(teds_sem, float) and math.isnan(teds_sem)) else "skip"

    print(
        f"\n[epoch {epoch:03d}/{EPOCHS}] lr={lr_now:.2e} | updates={updates} | train_loss={avg_train_loss:.4f}\n"
        f"  test_F1_all={f1_all_str} | test_F1_nonroot={f1_nr_str} | test_TEDS_sem={teds_str}{tag}\n"
        f"  epoch_time={epoch_time/60:.1f} min | eval_time={eval_time:.1f}s | ETA={eta/60:.1f} min\n"
    )

print("done. best_key=", best_key, "best_by=", BEST_BY, "saved:", SAVE_PATH)


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

# Final metrics on test_loader
p_all, r_all, f1_all = micro_prf(model_ssa, test_loader, exclude_root=False, root_id=ROOT_ID)
p_nr,  r_nr,  f1_nr  = micro_prf(model_ssa, test_loader, exclude_root=True,  root_id=ROOT_ID)
teds_sem = eval_semantic_teds(model_ssa, test_loader)

print("\n===== FINAL TEST REPORT =====")
print(f"F1_all     : {f1_all:.4f} (P={p_all:.4f}, R={r_all:.4f})")
print(f"F1_nonroot : {f1_nr:.4f} (P={p_nr:.4f}, R={r_nr:.4f})")
print(f"TEDS_sem   : {teds_sem:.4f}")

