from __future__ import annotations
from typing import Dict, Any

import time
import torch

from .losses import compute_losses, FocalLoss


import time


def train_one_epoch(model, loader, optimizer, cfg, focal_cls, focal_rel):
    device = next(model.parameters()).device

    model.train()
    logs = {"loss": [], "loss_cls": [], "loss_par": [], "loss_rel": []}

    start_time = time.time()
    last_print = start_time

    print(f"\n[Train] start epoch, num_docs = {len(loader)}")

    for step, doc in enumerate(loader):
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        # ===== forward =====
        out = model(doc)

        # ===== loss =====
        losses = compute_losses(out, doc, model.num_classes, model.num_rel,
                                cfg, focal_cls, focal_rel,device)

        # ===== backward =====
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ===== log =====
        for k in logs:
            logs[k].append(float(losses[k].cpu()))

        # ===== progress print =====
        if step == 0 or (step + 1) % 5 == 0:
            now = time.time()
            step_time = now - t0
            elapsed = now - start_time

            print(
                f"  step {step+1:4d}/{len(loader)} | "
                f"step_time={step_time:5.2f}s | "
                f"loss={logs['loss'][-1]:.4f} "
                f"(cls={logs['loss_cls'][-1]:.3f}, "
                f"par={logs['loss_par'][-1]:.3f}, "
                f"rel={logs['loss_rel'][-1]:.3f}) | "
                f"elapsed={elapsed/60:.1f}min"
            )

        # ===== quick sanity check (只在最前面几步) =====
        if step < 2:
            L = len(doc["units"])
            print(f"    sanity: seq_len={L}, "
                  f"avg_parent_candidates={sum(len(x) for x in out['par_logits'])/L:.1f}")

    epoch_time = time.time() - start_time
    print(f"[Train] epoch done in {epoch_time/60:.2f} min")

    return {k: sum(v)/max(len(v),1) for k,v in logs.items()}

def eval_one_epoch(model, loader, cfg, focal_cls, focal_rel):
    model.eval()
    logs = {"loss": [], "loss_cls": [], "loss_par": [], "loss_rel": []}

    for doc in loader:
        out = model(doc)
        losses = compute_losses(out, doc, model.num_classes, model.num_rel, cfg, focal_cls, focal_rel)
        for k in logs:
            logs[k].append(float(losses[k].cpu()))
    return {k: sum(v)/max(len(v),1) for k,v in logs.items()}
