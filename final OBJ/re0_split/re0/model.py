from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .embedders import SBERTTextEmbedder, LayoutPosPageEmbedder, VisualFPNRoIEmbedder


class DSPSModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_rel: int,
        M_cp: np.ndarray,
        cfg: CFG,
        use_text: bool = False,
        use_visual: bool = True,
        use_softmask: bool = True,
    ):
        super().__init__()
        self.use_softmask = use_softmask
        self.num_classes = num_classes
        self.num_rel = num_rel
        self.cfg = cfg
        self.use_text = use_text
        self.use_visual = use_visual

        # embeddings
        self.layout_pos_page = LayoutPosPageEmbedder(
            d_model=cfg.d_model,
            layout_bins=cfg.layout_bins,
            max_pos=cfg.max_1d_pos,
            max_pages=cfg.max_pages,
        )

        if use_text:
            self.text_emb = SBERTTextEmbedder(d_model=cfg.d_model)
        else:
            self.text_emb = None

        if use_visual:
            self.vis_emb = VisualFPNRoIEmbedder(d_model=cfg.d_model)
        else:
            self.vis_emb = None

        self.fuse_ln = nn.LayerNorm(cfg.d_model)

        # encoder (bidirectional)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        # subtask1: class
        self.cls_head = nn.Linear(cfg.d_model, num_classes)

        # decoder: structure-aware GRU
        self.gru = nn.GRU(input_size=cfg.d_model, hidden_size=cfg.d_model, batch_first=True)

        # attention projections for parent finding
        self.Wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.Wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # relation head (concat)
        self.rel_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, num_rel),
        )

        # store M_cp (torch)
        # shape: (num_classes+1, num_classes)   rows=parent_class + ROOT(row=num_classes), cols=child_class
        M = torch.tensor(M_cp, dtype=torch.float32)
        self.register_buffer("M_cp", M)

    def forward(self, doc: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        doc keys from your dataset:
          doc_id, units(list), y_cls(list), y_parent(list), y_rel(list), page_images(dict)
        returns logits:
          cls_logits: (L,C)
          par_logits: list of length L where par_logits[i] is (i,) logits for j in [0..i-1] + ROOT at index 0?
          rel_logits: (L, R) using GT parent during training convenience (可在 loss 里选)
        """
        units = doc["units"]
        L = len(units)
        doc_id = doc["doc_id"]
        page_images = doc["page_images"]

        # ---- embeddings sum: text + visual + layout/pos/page ----
        x = self.layout_pos_page(units, page_images)  # (L,d)

        if self.use_text:
            texts = [u.get("text","") for u in units]
            x = x + self.text_emb(doc_id, texts)

        if self.use_visual:
            x = x + self.vis_emb(units, page_images)

        x = self.fuse_ln(x)            # (L,d)
        x = x.unsqueeze(0)             # (1,L,d) for transformer batch_first

        # ---- encoder ----
        x_star = self.encoder(x)       # (1,L,d)
        x_star = x_star.squeeze(0)     # (L,d)

        # ---- class logits ----
        cls_logits = self.cls_head(x_star)  # (L,C)
        cls_prob = F.softmax(cls_logits, dim=-1)  # (L,C)

        # ---- ROOT representation ----
        root = x_star.mean(dim=0, keepdim=True)  # (1,d)

        # ---- GRU decoder (causal) ----
        # 论文写用 x*_{i-1} 驱动，这里用全序列输入 + 自己取 h_i（等价实现）
        h_seq, _ = self.gru(x_star.unsqueeze(0))  # (1,L,d)
        h_seq = h_seq.squeeze(0)                  # (L,d)

        # ---- parent logits with soft-mask ----
        # 我们把 candidate set 定义为 [ROOT] + [0..i-1]
        # 输出 par_logits[i] 形状 (i+1,) 其中 index0=ROOT, index k>0 对应 parent=j=k-1
        par_logits = []
        eps = 1e-8

        # 预先准备 ROOT 的 class distribution：用 uniform 或者用 mean prob；论文是扩展 P_cls(0) 为 (C+1)，ROOT=1
        # 我这里做：P_cls_root_over_parentclass = one-hot at ROOT row
        # 计算 P_dom 时使用 rows=parentclass+ROOT
        # 具体：P̃_cls(j) = [P_cls(j), 0] for real nodes；ROOT 用 [0..0,1]
        root_one = torch.zeros((1, self.num_classes + 1), device=x_star.device)
        root_one[0, self.num_classes] = 1.0

        # child prob 扩成 (C) 即原本；parent prob 扩成 (C+1)
        # 对每个 i:
        for i in range(L):
            q = self.Wq(h_seq[i:i+1])                # (1,d)
            # keys = [ROOT] + past h
            k_root = self.Wk(root)                   # (1,d)
            if i == 0:
                keys = k_root                        # (1,d)
            else:
                k_past = self.Wk(h_seq[:i])          # (i,d)
                keys = torch.cat([k_root, k_past], dim=0)  # (i+1,d)

            # dot-product scores
            score = (q @ keys.t()).squeeze(0)        # (i+1,)

            # ----- soft-mask prior: P_dom(i, j) -----
            # child distribution: (C)
            p_child = cls_prob[i:i+1]                # (1,C)
            # parent distributions:
            if i == 0:
                p_parent_ext = root_one              # (1,C+1)
            else:
                p_parent = cls_prob[:i]              # (i,C)
                zeros = torch.zeros((i,1), device=x_star.device)
                p_parent_ext = torch.cat([p_parent, zeros], dim=1)  # (i,C+1)
                p_parent_ext = torch.cat([root_one, p_parent_ext], dim=0)  # (i+1,C+1)

            # P_dom = p_parent_ext @ M_cp @ p_child^T
            # M_cp: (C+1,C)
            prior = (p_parent_ext @ self.M_cp @ p_child.t()).squeeze(-1)  # (i+1,)

            if self.use_softmask:
                score = score + torch.log(prior + eps)

            par_logits.append(score)

        # ---- relation logits (用 GT parent 保持训练稳定；推理时再用预测 parent) ----
        # 这里先输出一个 (L,R) 的 logits，其中第 i 个是与 GT parent 的关系
        # 对于 parent=-1 的（ROOT），relation 按你的数据里常见是 "contain"/"meta"，这里仍然算一个 rel loss（你也可 mask 掉）
        y_parent = doc.get("y_parent", [-1]*L)
        rel_logits = []
        for i in range(L):
            p = y_parent[i]
            if p is None or p < 0:
                # ROOT
                parent_vec = root.squeeze(0)
            else:
                parent_vec = h_seq[p]
            feat = torch.cat([h_seq[i], parent_vec], dim=-1)
            rel_logits.append(self.rel_head(feat))
        rel_logits = torch.stack(rel_logits, dim=0)  # (L,R)

        return {
            "cls_logits": cls_logits,
            "par_logits": par_logits,  # list of tensors
            "rel_logits": rel_logits,
        }
