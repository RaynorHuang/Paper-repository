from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N,C)
        targets: (N,)
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

def compute_losses(
    out: Dict[str, Any],
    doc: Dict[str, Any],
    num_classes: int,
    num_rel: int,
    cfg: CFG,
    focal_cls: nn.Module,
    focal_rel: nn.Module,
    device
) -> Dict[str, torch.Tensor]:

    # ---- targets ----
    y_cls = torch.tensor(doc["y_cls"], dtype=torch.long, device=device)
    y_parent = doc["y_parent"]
    y_rel = torch.tensor(doc["y_rel"], dtype=torch.long, device=device)

    # ---- logits ----
    cls_logits = out["cls_logits"].to(device)   # (L,C)
    rel_logits = out["rel_logits"].to(device)   # (L,R)
    par_logits_list = out["par_logits"]         # list length L

    L = len(par_logits_list)

    # ========== sanity checks ==========
    # cls range
    cls_min = int(y_cls.min().item())
    cls_max = int(y_cls.max().item())
    if cls_min < 0 or cls_max >= num_classes:
        raise ValueError(f"y_cls out of range: min={cls_min}, max={cls_max}, num_classes={num_classes}")

    # rel range
    rel_min = int(y_rel.min().item())
    rel_max = int(y_rel.max().item())
    if rel_min < 0 or rel_max >= num_rel:
        raise ValueError(f"y_rel out of range: min={rel_min}, max={rel_max}, num_rel={num_rel}")

    # parent range per position
    for i in range(L):
        p = y_parent[i]
        if p is None:
            continue
        if p >= i:  # 注意：parent 必须 < i（只能指向过去）
            raise ValueError(f"y_parent invalid at i={i}: parent={p} but must be < {i}")
        if p < -1:
            raise ValueError(f"y_parent invalid at i={i}: parent={p} (should be -1 or >=0)")

    # ========== losses ==========
    # meta mask：True 表示参与结构监督的单元（非 meta）
    is_meta = torch.tensor(doc.get("is_meta", [False]*L), dtype=torch.bool, device=device)
    struct_mask = ~is_meta  # (L,)

    # Subtask1: class（是否 mask meta 看你后续要不要对齐论文；先不 mask，保证能跑通）
    loss_cls = focal_cls(cls_logits, y_cls)

    # Subtask2: parent（只对非 meta 计算）
    loss_par = 0.0
    denom_par = 0
    for i, logits_i in enumerate(par_logits_list):
        if not bool(struct_mask[i].item()):
            continue

        p = y_parent[i]
        tgt_i = 0 if (p is None or p < 0) else (p + 1)

        if tgt_i < 0 or tgt_i >= logits_i.numel():
            raise ValueError(
                f"parent target out of range at i={i}: tgt={tgt_i}, "
                f"logits_len={logits_i.numel()}, raw_parent={p}"
            )

        tgt = torch.tensor([tgt_i], dtype=torch.long, device=device)
        # logits_i 可能是 shape=(K,) 或 (1,K)。统一压成 (1,K)
        logits_ce = logits_i.to(device).reshape(1, -1)
        loss_par = loss_par + F.cross_entropy(logits_ce, tgt)

        denom_par += 1

    if denom_par == 0:
        loss_par = torch.zeros((), device=device)
    else:
        loss_par = loss_par / denom_par

    # Subtask3: relation（只对非 meta 计算）
    if struct_mask.any():
        loss_rel = focal_rel(rel_logits[struct_mask], y_rel[struct_mask])
    else:
        loss_rel = torch.zeros((), device=device)

    loss = loss_cls + cfg.alpha_parent * loss_par + cfg.alpha_rel * loss_rel
    return {
        "loss": loss,
        "loss_cls": loss_cls.detach(),
        "loss_par": loss_par.detach(),
        "loss_rel": loss_rel.detach(),
    }
