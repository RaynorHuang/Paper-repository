from . import data_setup
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import json


MODEL_NAME = "microsoft/layoutlmv3-base"   
MAX_TOKENS = 512
USE_HRES = True


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class DocHieNetDocDataset(Dataset):
    def __init__(self, index_df, split="train"):
        self.df = index_df[index_df["split"] == split].reset_index(drop=True)
        self.root_id = 0

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
                "text": c.get("text", "") or "",
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
            text = (e["text"] or "").strip()
            if text == "":
                text = "[UNK]"

            
            words = text.split()
            if len(words) == 0:
                words = ["[UNK]"]

       
            word_boxes = [e["bbox"]] * len(words)

            enc = tokenizer(
                words,
                boxes=word_boxes,
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=False
            )

            ids = enc["input_ids"]
            am  = enc["attention_mask"]

            
            inner = list(range(len(ids)))

        
            bxs = [e["bbox"]] * len(ids)
            eids = [e["id"]]  * len(ids)
            pids = [e["page"]] * len(ids)

            if cur_len > 0 and cur_len + len(ids) > MAX_TOKENS:
                flush()

            if len(ids) > MAX_TOKENS:
                ids = ids[:MAX_TOKENS]; am = am[:MAX_TOKENS]
                inner = inner[:MAX_TOKENS]
                bxs = bxs[:MAX_TOKENS]; eids = eids[:MAX_TOKENS]; pids = pids[:MAX_TOKENS]

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
        from collections import OrderedDict
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
        from collections import OrderedDict
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
        parent_map = self._build_parent_map(elements)

        chunks = self._tokenize_and_chunk(elements)
        pooling = self._build_pooling(chunks)
        doc_elem_ids, doc_elem_positions = self._merge_doc_positions(chunks, pooling)

        return {
            "doc_id": doc_id,
            "chunks": chunks,
            "doc_elem_ids": doc_elem_ids,
            "doc_elem_positions": doc_elem_positions,
            "parent_map": parent_map,
        }

def collate_fn(batch):
   
    return batch[0]

train_ds = DocHieNetDocDataset(data_setup.index_df, split="train")
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

# quick iterate one batch
b = next(iter(train_loader))
print("doc_id:", b["doc_id"])
print("num chunks:", len(b["chunks"]))
print("num elements:", len(b["doc_elem_ids"]))


