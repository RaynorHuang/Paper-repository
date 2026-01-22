from . import data_setup
from .dataset_loader import DocHieNetDocDataset, collate_fn
import torch
from torch.utils.data import DataLoader
from . import model_ssa as model_ssa_mod

def edges_from_parent_map(doc_elem_ids, parent_map, root_id=0):
    return set((parent_map[eid], eid) for eid in doc_elem_ids)

def edges_from_logits(doc_elem_ids, logits, root_id=0):
    """
    logits: [M, M+1] where parents are [ROOT] + doc_elem_ids
    returns set of (parent_id, child_id)
    """
    parent_candidates = [root_id] + doc_elem_ids
    pred_parent_idx = torch.argmax(logits, dim=1).tolist() 
    pred_edges = set()
    for child_eid, pidx in zip(doc_elem_ids, pred_parent_idx):
        pred_parent_id = parent_candidates[pidx]
        pred_edges.add((pred_parent_id, child_eid))
    return pred_edges

def prf1(gt_edges, pred_edges):
    inter = gt_edges & pred_edges
    p = len(inter) / len(pred_edges) if pred_edges else 0.0
    r = len(inter) / len(gt_edges) if gt_edges else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    return p, r, f1, len(inter)


test_ds = DocHieNetDocDataset(data_setup.index_df, split="test")
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

model_ssa_mod.model_ssa.eval()
root_id = 0

tp_sum = 0
pred_sum = 0
gt_sum = 0


LIMIT_DOCS = None  

with torch.no_grad():
    for i, batch in enumerate(test_loader, start=1):
        loss, logits = model_ssa_mod.model_ssa(
            batch["chunks"],
            batch["doc_elem_ids"],
            batch["doc_elem_positions"],
            batch["parent_map"]
        )

        gt = edges_from_parent_map(batch["doc_elem_ids"], batch["parent_map"], root_id=root_id)
        pred = edges_from_logits(batch["doc_elem_ids"], logits, root_id=root_id)

        inter = gt & pred
        tp_sum += len(inter)
        pred_sum += len(pred)
        gt_sum += len(gt)

        if LIMIT_DOCS is not None and i >= LIMIT_DOCS:
            break


precision = tp_sum / pred_sum if pred_sum else 0.0
recall = tp_sum / gt_sum if gt_sum else 0.0
f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0

print(f"Evaluated docs: {i if LIMIT_DOCS is None else LIMIT_DOCS}")
print(f"TP={tp_sum}, Pred={pred_sum}, GT={gt_sum}")
print(f"Micro Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

