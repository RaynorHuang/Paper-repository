from __future__ import annotations
from typing import Dict, Any, List, Tuple

import os
import json
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from .model import DSPSModel
from .dataset import ID2LABEL_14, REL3_INV


@torch.no_grad()
def predict_doc(model: DSPSModel, doc: Dict[str, Any]) -> Dict[str, Any]:
    model.eval()
    out = model(doc)

    cls = out["cls_logits"].argmax(dim=-1).cpu().tolist()

    # parent: each i logits over [ROOT]+past
    par = []
    for i, logits_i in enumerate(out["par_logits"]):
        idx = int(torch.argmax(logits_i).cpu())
        p = -1 if idx == 0 else (idx - 1)
        par.append(p)

    # relation: 这里的 rel_logits 是用 GT parent 生成的；
    # 推理时更严格做法：用预测 parent 重算 rel（下面给你重算版本）
    # 先给简版：
    rel = out["rel_logits"].argmax(dim=-1).cpu().tolist()

    return {"pred_cls": cls, "pred_parent": par, "pred_rel": rel}


import json
from collections import defaultdict

@torch.no_grad()
def predict_doc_with_rel_recompute(model, doc, device):
    """
    返回：
      pred_cls: (L,)
      pred_parent: (L,)  -1 表示 ROOT
      pred_rel: (L,)
      also return raw probs/logits if needed
    """
    model.eval()
    out = model(doc)

    cls_logits = out["cls_logits"].to(device)          # (L,C)
    par_logits_list = out["par_logits"]                # list of (i+1,)
    L = cls_logits.shape[0]

    pred_cls = cls_logits.argmax(dim=-1).detach().cpu().tolist()

    # parent decode: candidates = [ROOT] + [0..i-1]
    pred_parent = []
    for i, logits_i in enumerate(par_logits_list):
        idx = int(torch.argmax(logits_i).item())
        p = -1 if idx == 0 else (idx - 1)
        pred_parent.append(p)

    # recompute relation logits using predicted parent
    # need access to h_seq and root inside the model; easiest: re-run minimal pieces here
    # We can reconstruct h_seq by running embeddings+encoder+gru again using model modules
    units = doc["units"]
    doc_id = doc["doc_id"]
    page_images = doc["page_images"]

    # embeddings sum (same as forward)
    x = model.layout_pos_page(units, page_images)
    if model.use_text:
        texts = [u.get("text","") for u in units]
        x = x + model.text_emb(doc_id, texts)
    if model.use_visual:
        x = x + model.vis_emb(units, page_images)
    x = model.fuse_ln(x)                       # (L,d)
    x_star = model.encoder(x.unsqueeze(0)).squeeze(0)  # (L,d)
    root = x_star.mean(dim=0, keepdim=True)    # (1,d)
    h_seq = model.gru(x_star.unsqueeze(0))[0].squeeze(0)  # (L,d)

    rel_logits = []
    for i in range(L):
        p = pred_parent[i]
        # --- safe parent index for relation recompute ---
        # p could be invalid (>=len) during early training; fallback to root for safety
        L = h_seq.size(0)
        if (p is None) or (p < 0) or (p >= L):
            parent_vec = root.squeeze(0)
            # 可选：记录一次非法 parent（不影响逻辑）
            # doc.setdefault("_bad_parent_count", 0)
                    # doc["_bad_parent_count"] += 1
        else:
            parent_vec = h_seq[p]

        feat = torch.cat([h_seq[i], parent_vec], dim=-1)
        rel_logits.append(model.rel_head(feat))

    rel_logits = torch.stack(rel_logits, dim=0)  # (L,R)

    pred_rel = rel_logits.argmax(dim=-1).detach().cpu().tolist()

    return {
        "pred_cls": pred_cls,
        "pred_parent": pred_parent,
        "pred_rel": pred_rel,
        "cls_logits": cls_logits.detach().cpu(),
        "rel_logits": rel_logits.detach().cpu(),
    }


def _id2name(mapping, idx: int) -> str:
    """
    mapping 可以是：
      - list: mapping[idx]
      - dict: mapping.get(idx)
      - None: 返回 idx 字符串
    """
    if mapping is None:
        return str(idx)
    if isinstance(mapping, dict):
        return mapping.get(idx, str(idx))
    if isinstance(mapping, (list, tuple)):
        if 0 <= idx < len(mapping):
            return str(mapping[idx])
        return str(idx)
    return str(idx)


def export_tree_json(doc, pred=None):
    """
    doc: dataset 返回的 doc dict
    pred: None => 导出 GT
          dict => 导出 pred（需要含 pred_cls/pred_parent/pred_rel）
    输出格式：
      {
        doc_id,
        nodes: [
          {id, text, label_id, label_name, is_meta, parent, rel_id, rel_name, page_id, box},
          ...
        ]
      }
    """
    doc_id = doc["doc_id"]
    units = doc["units"]
    L = len(units)

    if pred is None:
        cls_ids = doc["y_cls"]
        parent = doc["y_parent"]
        rel_ids = doc["y_rel"]
    else:
        cls_ids = pred["pred_cls"]
        parent = pred["pred_parent"]
        rel_ids = pred["pred_rel"]

    # 兼容你 notebook 里 ID2LABEL_14 / ID2REL 的类型（list 或 dict）
    label_map = globals().get("ID2LABEL_14", None)
    rel_map = globals().get("ID2REL", None)

    out = {"doc_id": doc_id, "nodes": []}
    is_meta_list = doc.get("is_meta", [False]*L)

    for i in range(L):
        u = units[i]
        is_meta = bool(is_meta_list[i])
        cls_id = int(cls_ids[i])
        rel_id = int(rel_ids[i]) if rel_ids is not None else -1

        out["nodes"].append({
            "id": i,
            "text": u.get("text", ""),
            "label_id": cls_id,
            "label_name": _id2name(label_map, cls_id),
            "is_meta": is_meta,
            "parent": int(parent[i]) if parent[i] is not None else -1,
            "rel_id": rel_id,
            "rel_name": _id2name(rel_map, rel_id),
            "page_id": int(u.get("page_id", 0)),
            "box": [int(x) for x in u.get("box", [0,0,0,0])]
        })
    return out


from tqdm import tqdm
import os

@torch.no_grad()
def export_split_predictions(model, loader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)

    for doc in tqdm(loader, desc=f"export -> {save_dir}"):
        pred = predict_doc_with_rel_recompute(model, doc, device)

        gt_json = export_tree_json(doc, pred=None)
        pr_json = export_tree_json(doc, pred=pred)

        doc_id = doc["doc_id"].replace("/", "_")
        with open(os.path.join(save_dir, f"{doc_id}.gt.json"), "w", encoding="utf-8") as f:
            json.dump(gt_json, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, f"{doc_id}.pred.json"), "w", encoding="utf-8") as f:
            json.dump(pr_json, f, ensure_ascii=False, indent=2)

    print("done.")
