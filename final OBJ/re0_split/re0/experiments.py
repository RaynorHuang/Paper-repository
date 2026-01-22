from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import os
import json
import time
import numpy as np
import torch
import inspect

from .model import DSPSModel
from .losses import FocalLoss
from .train import train_one_epoch
from .predict import export_split_predictions
from .metrics import eval_export_dir


def check_causal_violation(dataset, n=100):
    cnt = 0
    viol = 0
    for i in range(min(n, len(dataset))):
        d = dataset[i]
        yp = d["y_parent"]
        for j,p in enumerate(yp):
            if p is not None and p >= j:
                viol += 1
            cnt += 1
    return viol / max(cnt,1)



import numpy as np
import json, os, glob

@torch.no_grad()
def eval_export_dir_dual(export_dir: str, exclude_meta: bool = True,show_progress: bool = True):
    pairs = load_pair_files(export_dir)

    st_strict, st_label = [], []
    cls_acc, par_acc, rel_acc = [], [], []

    for gt_path, pr_path in pairs:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_js = json.load(f)
        with open(pr_path, "r", encoding="utf-8") as f:
            pr_js = json.load(f)

        st_strict.append(compute_steds(gt_js, pr_js, exclude_meta=exclude_meta, match_mode="strict"))
        st_label.append(compute_steds(gt_js, pr_js, exclude_meta=exclude_meta, match_mode="label"))

        aux = compute_aux_metrics(gt_js, pr_js, exclude_meta=exclude_meta)
        cls_acc.append(aux["cls_acc"])
        par_acc.append(aux["parent_acc"])
        rel_acc.append(aux["rel_acc"])

    def agg(x):
        x = np.array(x, dtype=float)
        return float(x.mean()), float(x.std()), float(np.median(x))

    s_mean, s_std, s_med = agg(st_strict)
    l_mean, l_std, l_med = agg(st_label)

    return {
        "num_docs": len(pairs),
        "STEDS_strict_mean": s_mean,
        "STEDS_strict_std": s_std,
        "STEDS_strict_median": s_med,
        "STEDS_label_mean": l_mean,
        "STEDS_label_std": l_std,
        "STEDS_label_median": l_med,
        "cls_acc_mean": float(np.mean(cls_acc)) if cls_acc else 0.0,
        "parent_acc_mean": float(np.mean(par_acc)) if par_acc else 0.0,
        "rel_acc_mean": float(np.mean(rel_acc)) if rel_acc else 0.0,
    }


import time


def run_experiment(exp_name: str, use_softmask: bool, train_loader, test_loader, M_cp, cfg, seed=42, save_root="ablation_runs"):
    # 固定随机种子
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    exp_dir = os.path.join(save_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 建模：只改变 use_softmask
    model = DSPSModel(
        num_classes=len(ID2LABEL_14),
        num_rel=len(REL2ID),
        M_cp=M_cp,
        cfg=None,
        use_text=True,
        use_visual= False,
        use_softmask=use_softmask,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    focal_cls = FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha).to(device)
    focal_rel = FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha).to(device)

    best = 1e18
    best_path = os.path.join(exp_dir, "best.pt")

    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, cfg, focal_cls, focal_rel)
        te = eval_one_epoch(model, test_loader, cfg, focal_cls, focal_rel)
        print(f"[{exp_name}] ep={ep} train={tr} test={te}")

        if te["loss"] < best:
            best = te["loss"]
            torch.save(model.state_dict(), best_path)
            print(f"[{exp_name}] saved -> {best_path}")

    # 导出与评测
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    export_dir = os.path.join(exp_dir, "exports_test")
    export_split_predictions(model, test_loader, save_dir=export_dir, device=device)

    print(f"[{exp_name}] export done, start eval...")
    metrics = eval_export_dir_dual(export_dir, exclude_meta=True, show_progress=True)
    metrics.update({
        "exp_name": exp_name,
        "use_softmask": use_softmask,
        "seed": seed,
        "epochs": cfg.epochs,
        "elapsed_sec": round(time.time() - t0, 2),
        "export_dir": export_dir,
    })
    print(f"[{exp_name}] eval done.")


    return metrics




def _call_with_compatible_signature(fn, *args, **kwargs):
    """
    Call fn with kwargs filtered to those supported by fn's signature.
    This prevents errors like: got an unexpected keyword argument 'device'.
    """
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(*args, **filtered)



def train_one_epoch_compat(model, loader, optimizer, *, device=None, cfg=None, scaler=None):
        if focal_cls is None:
            focal_cls = FocalLoss(gamma=getattr(cfg, "focal_gamma", 2.0),
                                  alpha=getattr(cfg, "focal_alpha", 0.25)).to(device)
        if focal_rel is None:
            focal_rel = FocalLoss(gamma=getattr(cfg, "focal_gamma", 2.0),
                                  alpha=getattr(cfg, "focal_alpha", 0.25)).to(device)
    # Try calling with keywords first; incompatible keywords are filtered out.
        return _call_with_compatible_signature(
        train_one_epoch,
        model, loader, optimizer,
        device=device, cfg=None, scaler=scaler
    )
