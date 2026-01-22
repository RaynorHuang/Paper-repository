from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class CFG:
    # data
    max_len: int = 512
    batch_size: int = 1          # doc-level batch=1（结构任务依赖整篇文档序列）
    num_workers: int = 2

    # model dims
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.1

    # embedding vocab sizes
    max_1d_pos: int = 512
    max_pages: int = 32

    # layout emb
    box_bins: int = 1000
    layout_bins: int = 1001  
    page_emb_dim: int = 64

    # text emb
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_out_dim: int = 384
    text_proj_dim: int = 256

    # visual emb (only used when use_visual=True)
    vis_backbone: str = "resnet50"
    vis_crop_size: int = 224
    vis_out_dim: int = 256

    # losses
    alpha_parent: float = 1.0
    alpha_rel: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # optim
    lr: float = 2e-4
    weight_decay: float = 1e-2
    epochs: int = 10

def normalize_box_xyxy(box, page_w: float, page_h: float, bins: int = 1000):
    """像素坐标 [x0,y0,x1,y1] -> 整数归一化到 [0,bins]."""
    x0, y0, x1, y1 = box
    x0 = int(np.clip(round(x0 / max(page_w, 1) * bins), 0, bins))
    x1 = int(np.clip(round(x1 / max(page_w, 1) * bins), 0, bins))
    y0 = int(np.clip(round(y0 / max(page_h, 1) * bins), 0, bins))
    y1 = int(np.clip(round(y1 / max(page_h, 1) * bins), 0, bins))
    return [x0, y0, x1, y1]

def collate_doc(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(batch) == 1, "当前实现是 doc-level batch=1"
    return batch[0]
