from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from .config import normalize_box_xyxy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class SBERTTextEmbedder(nn.Module):

    def __init__(self, d_model: int, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.d_model = d_model
        self.model_name = model_name
        
        self._sbert = None
        self.proj = nn.Linear(384, d_model)  # all-MiniLM-L6-v2 输出 384
        self.cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def _lazy_load(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.model_name)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self._lazy_load()
        emb = self._sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False)  # (L,384)
        return emb

    def forward(self, doc_id: str, texts: List[str]) -> torch.Tensor:
        
        device = self.proj.weight.device
        L = len(texts)
        out = [None] * L
        missing_idx, missing_text = [], []

        # 1) cache hit：取 CPU -> to(device)
        for i, t in enumerate(texts):
            key = (doc_id, i)
            if key in self.cache:
                out[i] = self.cache[key].to(device)
            else:
                missing_idx.append(i)
                missing_text.append(t)


        if len(missing_idx) > 0:
            emb_384 = self.encode_texts(missing_text)          
            emb_384 = emb_384.detach().clone().to(device)      
            emb_d = self.proj(emb_384)                         

            for k, i in enumerate(missing_idx):
                key = (doc_id, i)
                self.cache[key] = emb_d[k].detach().cpu()      
                out[i] = emb_d[k]                              

        return torch.stack(out, dim=0)
    

class LayoutPosPageEmbedder(nn.Module):
    def __init__(self, d_model: int, layout_bins: int = 1001, max_pos: int = 512, max_pages: int = 32):
        super().__init__()
        self.d_model = d_model
        self.layout_bins = layout_bins

        
        self.emb_x0 = nn.Embedding(layout_bins, d_model//4)
        self.emb_x1 = nn.Embedding(layout_bins, d_model//4)
        self.emb_w  = nn.Embedding(layout_bins, d_model//4)
        self.emb_y0 = nn.Embedding(layout_bins, d_model//4)
        self.emb_y1 = nn.Embedding(layout_bins, d_model//4)
        self.emb_h  = nn.Embedding(layout_bins, d_model//4)

        self.proj_layout = nn.Linear((d_model//4)*6, d_model)

        self.emb_pos = nn.Embedding(max_pos, d_model)
        self.emb_page = nn.Embedding(max_pages, d_model)

        self.ln = nn.LayerNorm(d_model)

    def forward(self, units: List[Dict[str, Any]], page_images: Dict[int, str]) -> torch.Tensor:
        """
        return: (L, d_model)
        """
        L = len(units)
        
        page_wh = {}
        for pid, pth in page_images.items():
            img = Image.open(pth)
            page_wh[int(pid)] = (img.width, img.height)

        layout_vecs = []
        pos_ids = []
        page_ids = []

        for i,u in enumerate(units):
            pid = int(u["page_id"])
            w, h = page_wh[pid]
            b = u.get("box", None)
            if b is None:
                b = u.get("bbox", None)
            if b is None:
                raise KeyError("Unit missing both 'box' and 'bbox'")
            x0, y0, x1, y1 = b

            nb = normalize_box_xyxy([x0,y0,x1,y1], w, h, bins=self.layout_bins-1)
            nx0, ny0, nx1, ny1 = nb
            nw = int(np.clip(nx1 - nx0, 0, self.layout_bins-1))
            nh = int(np.clip(ny1 - ny0, 0, self.layout_bins-1))

            layout_vecs.append([nx0, nx1, nw, ny0, ny1, nh])
            pos_ids.append(i)
            page_ids.append(min(pid, self.emb_page.num_embeddings-1))

        layout = torch.tensor(layout_vecs, dtype=torch.long, device=self.emb_pos.weight.device)  # (L,6)
        pos = torch.tensor(pos_ids, dtype=torch.long, device=self.emb_pos.weight.device)        # (L,)
        pages = torch.tensor(page_ids, dtype=torch.long, device=self.emb_pos.weight.device)     # (L,)

        x0 = self.emb_x0(layout[:,0])
        x1 = self.emb_x1(layout[:,1])
        ww = self.emb_w(layout[:,2])
        y0 = self.emb_y0(layout[:,3])
        y1 = self.emb_y1(layout[:,4])
        hh = self.emb_h(layout[:,5])
        layout_cat = torch.cat([x0,x1,ww,y0,y1,hh], dim=-1)
        layout_emb = self.proj_layout(layout_cat)

        pos_emb = self.emb_pos(pos)
        page_emb = self.emb_page(pages)

        out = self.ln(layout_emb + pos_emb + page_emb)
        return out

import torch
import torch.nn as nn
from PIL import Image


class VisualFPNRoIEmbedder(nn.Module):
    """
    Paper-style visual embedding:
    page image -> ResNet50+FPN -> RoIAlign by bbox -> pooled -> Linear -> (L, d_model)

    Interface kept identical to your existing VisualCropEmbedder:
      forward(units, page_images) -> Tensor[L, d_model]
    where:
      - units: list of dict, each must have:
          u["page_id"] : int
          u["box"]     : [x0, y0, x1, y1] in *pixel coords of the original page image*
      - page_images: dict or list-like indexed by page_id -> image path
    """
    def __init__(
        self,
        d_model: int,
        roi_out_size: int = 7,     # RoIAlign output spatial size (7x7 typical)
        roi_sampling_ratio: int = 2,
        fpn_level: str = "0",      # torchvision FPN returns keys like "0","1","2","3","pool"
    ):
        super().__init__()
        self.d_model = d_model
        self.roi_out_size = roi_out_size
        self.roi_sampling_ratio = roi_sampling_ratio
        self.fpn_level = fpn_level

        # ResNet50 + FPN backbone (detection-style)
        # returns dict of feature maps: {"0":P2,"1":P3,"2":P4,"3":P5,"pool":P6}
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=torchvision.models.ResNet50_Weights.DEFAULT,
            trainable_layers=3,   

        # For resnet_fpn_backbone, each pyramid level channel is 256
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, d_model)

        # ImageNet normalization
        w = torchvision.models.ResNet50_Weights.DEFAULT
        mean, std = w.transforms().mean, w.transforms().std
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _load_page(self, page_path: str):
        img = Image.open(page_path).convert("RGB")
        return img

    def _to_tensor(self, pil_img: Image.Image, device: torch.device):
        x = self.tf(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)
        return x

    @staticmethod
    def _boxes_to_roi_format(boxes_xyxy, batch_idx: int = 0, device=None):
        """
        roi_align expects boxes as Tensor[K,5] = (batch_idx, x1, y1, x2, y2)
        in input image pixel coordinates when spatial_scale is set properly.
        """
        b = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device)
        if b.numel() == 0:
            return b.new_zeros((0, 5))
        idx = torch.full((b.shape[0], 1), float(batch_idx), device=device)
        return torch.cat([idx, b], dim=1)

    def forward(self, units, page_images):
        device = self.proj.weight.device

        # group unit indices by page_id (so each page runs backbone once)
        page_to_indices = {}
        for i, u in enumerate(units):
            pid = int(u["page_id"])
            page_to_indices.setdefault(pid, []).append(i)

        out = torch.zeros((len(units), self.d_model), device=device)

        for pid, idxs in page_to_indices.items():
            page_path = page_images[pid]
            pil_img = self._load_page(page_path)
            W, H = pil_img.size

            x = self._to_tensor(pil_img, device=device)  # (1,3,H,W)

            feats = self.backbone(x)  # dict of feature maps
            if self.fpn_level not in feats:
                # fallback: use the highest-resolution level if key missing
                # typical keys: "0","1","2","3","pool"
                key = sorted([k for k in feats.keys() if k != "pool"])[0]
            else:
                key = self.fpn_level

            fmap = feats[key]  # (1,256,hf,wf)
            hf, wf = fmap.shape[-2], fmap.shape[-1]

            # spatial_scale maps original image coords -> feature map coords
            # roi_align uses a single scalar; assume isotropic scaling (works because fmap came from that image)
            spatial_scale = wf / float(W)

            boxes_xyxy = [units[i]["box"] for i in idxs]  # list of [x0,y0,x1,y1] in image pixels
            rois = self._boxes_to_roi_format(boxes_xyxy, batch_idx=0, device=device)  # (K,5)

            roi_feat = roi_align(
                input=fmap,
                boxes=rois,
                output_size=(self.roi_out_size, self.roi_out_size),
                spatial_scale=spatial_scale,
                sampling_ratio=self.roi_sampling_ratio,
                aligned=True,
            )  # (K,256,roi,roi)

            roi_feat = self.pool(roi_feat).flatten(1)  # (K,256)
            emb = self.proj(roi_feat)                  # (K,d_model)

            out[idxs] = emb

        return out
