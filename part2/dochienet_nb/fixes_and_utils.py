import types
import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

def sinusoidal_position_encoding(pos, dim):
    pos = pos.float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=pos.device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(pos.size(0), dim, device=pos.device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

def _encode_one_chunk_clamped(self, c):
    input_ids = torch.tensor(c["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(c["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)

    bbox = torch.tensor(c["bbox"], dtype=torch.long, device=device).unsqueeze(0)
    bbox = torch.clamp(bbox, 0, 1000)  # <<<关键修复：保证在 [0,1000]

    o = self.encoder(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, return_dict=True)
    hidden = o.last_hidden_state.squeeze(0)  # [T,H]

    # 下面保持与你原来的 DHFormerWithSSA 一致：page + inner embedding injection
    page_ids = torch.tensor(c["page_id"], dtype=torch.long, device=device)
    inner_pos = torch.tensor(c["inner_pos"], dtype=torch.long, device=device)

    pe = sinusoidal_position_encoding(page_ids, self.page_pe_dim)
    page_e = self.page_proj(pe)

    inner_pos = torch.clamp(inner_pos, 0, self.inner_emb.num_embeddings - 1)
    inner_e = self.inner_emb(inner_pos)

    return hidden + page_e + inner_e

model_ssa._encode_one_chunk = types.MethodType(_encode_one_chunk_clamped, model_ssa)

print("Patched model_ssa: bbox will be clamped to [0,1000] before encoder.")


from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import json
from collections import OrderedDict

MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_TOKENS = 512
MAX_CHUNKS_TRAIN = 32
MAX_CHUNKS_TEST  = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class DocHieNetDocDataset(Dataset):
    def __init__(self, index_df, split="train"):
        self.df = index_df[index_df["split"] == split].reset_index(drop=True)
        self.root_id = 0
        self.split = split
        self.max_chunks = MAX_CHUNKS_TRAIN if split == "train" else (MAX_CHUNKS_TEST if split == "test" else None)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _box_to_1000(box, pw, ph):
        x0, y0, x1, y1 = box
        return [
            int(1000 * x0 / pw),
            int(1000 * y0 / ph),
            int(1000 * x1 / pw),
            int(1000 * y1 / ph),
        ]

    def _load_elements(self, label_path):
        obj = json.loads(Path(label_path).read_text(encoding="utf-8"))
        pages_meta = obj["pages"]
        contents = sorted(obj["contents"], key=lambda x: x.get("order", 10**9))

        elements = []
        for c in contents:
            p = c["page"]
            pw, ph = pages_meta[f"page{p}"]["width"], pages_meta[f"page{p}"]["height"]
            elements.append({
                "id": c["id"],
                "page": p,
                "text": (c.get("text", "") or ""),
                "bbox": self._box_to_1000(c["box"], pw, ph),
                "linking": c.get("linking", []),
                "order": c.get("order", -1),
            })
        return elements

    def _build_parent_map(self, elements):
        parent_map = {}
        for e in elements:
            for pair in e.get("linking", []):
                if isinstance(pair, list) and len(pair) == 2:
                    p, c = pair
                    if c not in parent_map:
                        parent_map[c] = p
        for e in elements:
            cid = e["id"]
            if cid not in parent_map:
                parent_map[cid] = self.root_id
            if parent_map[cid] == -1:
                parent_map[cid] = self.root_id
        return parent_map

    def _tokenize_and_chunk(self, elements):
        """
        LayoutLMv3 tokenizer requires:
          - words: List[str]
          - boxes: List[List[int]] (one box per word)
        We'll assign each word the element bbox.
        """
        chunks = []
        cur = {"input_ids": [], "attention_mask": [], "bbox": [], "elem_id": [], "page_id": [], "inner_pos": []}
        cur_len = 0

        def flush():
            nonlocal cur, cur_len
            if cur_len > 0:
                chunks.append(cur)
            cur = {"input_ids": [], "attention_mask": [], "bbox": [], "elem_id": [], "page_id": [], "inner_pos": []}
            cur_len = 0

        for e in elements:
            text = e["text"].strip()
            words = text.split() if text else []
            if len(words) == 0:
                words = ["[UNK]"]

            boxes = [e["bbox"]] * len(words)

            enc = tokenizer(
                words,
                boxes=boxes,
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=False
            )

            ids = enc["input_ids"]
            am  = enc["attention_mask"]
            bxs = enc["bbox"]  # token-level bboxes aligned by tokenizer

            # inner position within this element (token-level)
            inner = list(range(len(ids)))
            eids = [e["id"]] * len(ids)
            pids = [e["page"]] * len(ids)

            if cur_len > 0 and cur_len + len(ids) > MAX_TOKENS:
                flush()

            # safeguard: element too long
            if len(ids) > MAX_TOKENS:
                ids = ids[:MAX_TOKENS]; am = am[:MAX_TOKENS]; bxs = bxs[:MAX_TOKENS]
                inner = inner[:MAX_TOKENS]; eids = eids[:MAX_TOKENS]; pids = pids[:MAX_TOKENS]

            cur["input_ids"].extend(ids)
            cur["attention_mask"].extend(am)
            cur["bbox"].extend(bxs)
            cur["elem_id"].extend(eids)
            cur["page_id"].extend(pids)
            cur["inner_pos"].extend(inner)
            cur_len += len(ids)

        flush()
        return chunks

    @staticmethod
    def _build_pooling(chunks):
        pooling = []
        for c in chunks:
            seen = OrderedDict()
            for i, eid in enumerate(c["elem_id"]):
                if eid not in seen:
                    seen[eid] = i
            pooling.append({"elem_ids": list(seen.keys()), "first_token_idx": list(seen.values())})
        return pooling

    @staticmethod
    def _merge_doc_positions(chunks, pooling):
        pos = OrderedDict()
        for chunk_idx, pool in enumerate(pooling):
            for eid, tok_idx in zip(pool["elem_ids"], pool["first_token_idx"]):
                if eid not in pos:
                    pos[eid] = (chunk_idx, tok_idx)
        doc_elem_ids = list(pos.keys())
        return doc_elem_ids, pos

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        doc_id = row["doc_id"]
        label_path = row["label_path"]

        elements = self._load_elements(label_path)
        parent_map_full = self._build_parent_map(elements)

        chunks = self._tokenize_and_chunk(elements)

        # cap chunks per paper setting
        if self.max_chunks is not None and len(chunks) > self.max_chunks:
            chunks = chunks[:self.max_chunks]

        pooling = self._build_pooling(chunks)
        doc_elem_ids, doc_elem_positions = self._merge_doc_positions(chunks, pooling)

        parent_map = {eid: parent_map_full[eid] for eid in doc_elem_ids}

        return {
            "doc_id": doc_id,
            "chunks": chunks,
            "doc_elem_ids": doc_elem_ids,
            "doc_elem_positions": doc_elem_positions,
            "parent_map": parent_map,
            "n_chunks": len(chunks),
            "n_elems": len(doc_elem_ids),
        }

def collate_fn(batch):
    return batch[0]

train_ds = DocHieNetDocDataset(index_df, split="train")
test_ds  = DocHieNetDocDataset(index_df, split="test")

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

b = next(iter(train_loader))
print("train sample:", b["doc_id"], "| chunks:", b["n_chunks"], "| elems:", b["n_elems"])
b2 = next(iter(test_loader))
print("test sample: ", b2["doc_id"], "| chunks:", b2["n_chunks"], "| elems:", b2["n_elems"])


# === Notebook cell 17 ===
# CLEANUP CELL: run once before training to avoid GPU memory fragmentation

import gc
import torch

for name in ["model", "model_pe", "model_dec", "encoder", "model_ssa_old", "model_dec_old"]:
    if name in globals():
        del globals()[name]

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("cleanup done")

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import json
from collections import OrderedDict

MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_TOKENS = 512

MAX_CHUNKS_TRAIN = 16   # e.g., 16 for RTX 4060 8GB
MAX_CHUNKS_TEST  = 64   # e.g., 64 (can raise later)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class DocHieNetDocDataset(Dataset):
    def __init__(self, index_df, split="train"):
        self.df = index_df[index_df["split"] == split].reset_index(drop=True)
        self.root_id = 0
        self.split = split
        if split == "train":
            self.max_chunks = MAX_CHUNKS_TRAIN
        elif split == "test":
            self.max_chunks = MAX_CHUNKS_TEST
        else:
            self.max_chunks = None

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _box_to_1000(box, pw, ph):
        x0, y0, x1, y1 = box
        return [
            int(1000 * x0 / pw),
            int(1000 * y0 / ph),
            int(1000 * x1 / pw),
            int(1000 * y1 / ph),
        ]

    def _load_elements(self, label_path):
        obj = json.loads(Path(label_path).read_text(encoding="utf-8"))
        pages_meta = obj["pages"]
        contents = sorted(obj["contents"], key=lambda x: x.get("order", 10**9))
        elements = []
        for c in contents:
            p = c["page"]
            pw, ph = pages_meta[f"page{p}"]["width"], pages_meta[f"page{p}"]["height"]
            elements.append({
                "id": c["id"],
                "page": p,
                "text": (c.get("text", "") or ""),
                "bbox": self._box_to_1000(c["box"], pw, ph),
                "linking": c.get("linking", []),
                "order": c.get("order", -1),
            })
        return elements

    def _build_parent_map(self, elements):
        parent_map = {}
        for e in elements:
            for pair in e.get("linking", []):
                if isinstance(pair, list) and len(pair) == 2:
                    p, c = pair
                    if c not in parent_map:
                        parent_map[c] = p
        for e in elements:
            cid = e["id"]
            if cid not in parent_map:
                parent_map[cid] = self.root_id
            if parent_map[cid] == -1:
                parent_map[cid] = self.root_id
        return parent_map

    def _tokenize_and_chunk(self, elements):
        chunks = []
        cur = {"input_ids": [], "attention_mask": [], "bbox": [], "elem_id": [], "page_id": [], "inner_pos": []}
        cur_len = 0

        def flush():
            nonlocal cur, cur_len
            if cur_len > 0:
                chunks.append(cur)
            cur = {"input_ids": [], "attention_mask": [], "bbox": [], "elem_id": [], "page_id": [], "inner_pos": []}
            cur_len = 0

        for e in elements:
            text = e["text"].strip()
            words = text.split() if text else []
            if len(words) == 0:
                words = ["[UNK]"]
            boxes = [e["bbox"]] * len(words)

            enc = tokenizer(
                words,
                boxes=boxes,
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=False
            )

            ids = enc["input_ids"]
            am  = enc["attention_mask"]
            bxs = enc["bbox"]

            inner = list(range(len(ids)))
            eids = [e["id"]] * len(ids)
            pids = [e["page"]] * len(ids)

            if cur_len > 0 and cur_len + len(ids) > MAX_TOKENS:
                flush()

            if len(ids) > MAX_TOKENS:
                ids = ids[:MAX_TOKENS]; am = am[:MAX_TOKENS]; bxs = bxs[:MAX_TOKENS]
                inner = inner[:MAX_TOKENS]; eids = eids[:MAX_TOKENS]; pids = pids[:MAX_TOKENS]

            cur["input_ids"].extend(ids)
            cur["attention_mask"].extend(am)
            cur["bbox"].extend(bxs)
            cur["elem_id"].extend(eids)
            cur["page_id"].extend(pids)
            cur["inner_pos"].extend(inner)
            cur_len += len(ids)

        flush()
        return chunks

    @staticmethod
    def _build_pooling(chunks):
        pooling = []
        for c in chunks:
            seen = OrderedDict()
            for i, eid in enumerate(c["elem_id"]):
                if eid not in seen:
                    seen[eid] = i
            pooling.append({"elem_ids": list(seen.keys()), "first_token_idx": list(seen.values())})
        return pooling

    @staticmethod
    def _merge_doc_positions(chunks, pooling):
        pos = OrderedDict()
        for chunk_idx, pool in enumerate(pooling):
            for eid, tok_idx in zip(pool["elem_ids"], pool["first_token_idx"]):
                if eid not in pos:
                    pos[eid] = (chunk_idx, tok_idx)
        doc_elem_ids = list(pos.keys())
        return doc_elem_ids, pos

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        doc_id = row["doc_id"]
        label_path = row["label_path"]

        elements = self._load_elements(label_path)
        parent_map_full = self._build_parent_map(elements)

        chunks = self._tokenize_and_chunk(elements)

        # cap chunks
        if self.max_chunks is not None and len(chunks) > self.max_chunks:
            chunks = chunks[:self.max_chunks]

        pooling = self._build_pooling(chunks)
        doc_elem_ids, doc_elem_positions = self._merge_doc_positions(chunks, pooling)

        kept = set(doc_elem_ids)
        root = self.root_id

        # --- FIX: if parent not in kept, remap to ROOT ---
        parent_map = {}
        for eid in doc_elem_ids:
            p = parent_map_full.get(eid, root)
            if p == -1:
                p = root
            if p != root and p not in kept:
                p = root
            if p == eid:  # guard self-loop
                p = root
            parent_map[eid] = p

        return {
            "doc_id": doc_id,
            "chunks": chunks,
            "doc_elem_ids": doc_elem_ids,
            "doc_elem_positions": doc_elem_positions,
            "parent_map": parent_map,
            "n_chunks": len(chunks),
            "n_elems": len(doc_elem_ids),
        }

def collate_fn(batch):
    return batch[0]

train_ds = DocHieNetDocDataset(index_df, split="train")
test_ds  = DocHieNetDocDataset(index_df, split="test")
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

b = next(iter(train_loader))
print("sample:", b["doc_id"], "| chunks:", b["n_chunks"], "| elems:", b["n_elems"])
print("parents missing after fix:",
      sum(1 for eid in b["doc_elem_ids"] if b["parent_map"][eid] not in set([0]+b["doc_elem_ids"])))


