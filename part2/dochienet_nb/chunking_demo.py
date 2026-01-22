import json
from pathlib import Path
from collections import OrderedDict

from transformers import AutoTokenizer

# notebook 全局状态统一从 data_setup 来
from . import data_setup


MODEL_NAME = "bert-base-uncased"
MAX_TOKENS = 512
DOC_ID = "0a51k4mj12"

DATA_DIR = data_setup.DATA_DIR


# ---------- load label ----------
label_path = DATA_DIR / "labels" / f"{DOC_ID}.json"
obj = json.loads(label_path.read_text(encoding="utf-8"))
pages_meta = obj["pages"]
contents = sorted(obj["contents"], key=lambda x: x.get("order", 10**9))


def box_to_1000(box, page_w, page_h):
    x0, y0, x1, y1 = box
    return [
        int(1000 * x0 / page_w),
        int(1000 * y0 / page_h),
        int(1000 * x1 / page_w),
        int(1000 * y1 / page_h),
    ]


elements = []
for c in contents:
    p = c["page"]
    pw, ph = pages_meta[f"page{p}"]["width"], pages_meta[f"page{p}"]["height"]
    elements.append({
        "id": c["id"],
        "page": p,
        "label": c.get("label", ""),
        "text": c.get("text", ""),
        "bbox": box_to_1000(c["box"], pw, ph),
        "linking": c.get("linking", []),
        "order": c.get("order", -1),
    })


def build_parent_map(elements, root_id=0):
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
            parent_map[cid] = root_id
        if parent_map[cid] == -1:
            parent_map[cid] = root_id
    return parent_map


parent_map = build_parent_map(elements, root_id=0)


# ---------- tokenizer ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

chunks = []
cur = {
    "input_ids": [],
    "attention_mask": [],
    "bbox": [],
    "page_id": [],
    "elem_id": [],
    "inner_pos": [],
}
cur_len = 0
nonlocal_chunks = []

for e in elements:
    text = e["text"] if e["text"] is not None else ""
    if text.strip() == "":
        text = "[UNK]"

    enc = tok(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=False,
        truncation=False,
    )

    ids = enc["input_ids"]
    am = enc["attention_mask"]
    inner = list(range(len(ids)))

    bxs = [e["bbox"]] * len(ids)
    pids = [e["page"]] * len(ids)
    eids = [e["id"]] * len(ids)

    if cur_len > 0 and cur_len + len(ids) > MAX_TOKENS:
        nonlocal_chunks.append(cur)
        cur = {
            "input_ids": [],
            "attention_mask": [],
            "bbox": [],
            "page_id": [],
            "elem_id": [],
            "inner_pos": [],
        }
        cur_len = 0

    if len(ids) > MAX_TOKENS:
        ids = ids[:MAX_TOKENS]
        am = am[:MAX_TOKENS]
        inner = inner[:MAX_TOKENS]
        bxs = bxs[:MAX_TOKENS]
        pids = pids[:MAX_TOKENS]
        eids = eids[:MAX_TOKENS]

    cur["input_ids"].extend(ids)
    cur["attention_mask"].extend(am)
    cur["bbox"].extend(bxs)
    cur["page_id"].extend(pids)
    cur["elem_id"].extend(eids)
    cur["inner_pos"].extend(inner)
    cur_len += len(ids)

if cur_len > 0:
    nonlocal_chunks.append(cur)

chunks = nonlocal_chunks


print("DOC_ID:", DOC_ID)
print("num elements:", len(elements))
print("num chunks:", len(chunks))
print("chunk lens (first 10):", [len(c["input_ids"]) for c in chunks[:10]])
print("max chunk len:", max(len(c["input_ids"]) for c in chunks))


# ---------- pooling ----------
def build_pooling_indices(chunks):
    pooling = []

    for chunk in chunks:
        seen = OrderedDict()
        for idx, eid in enumerate(chunk["elem_id"]):
            if eid not in seen:
                seen[eid] = idx

        pooling.append({
            "elem_ids": list(seen.keys()),
            "first_token_idx": list(seen.values())
        })

    return pooling


pooling = build_pooling_indices(chunks)

print("num chunks:", len(pooling))

for i in range(min(3, len(pooling))):
    p = pooling[i]
    print(f"\nChunk {i}:")
    print("  num elements in chunk:", len(p["elem_ids"]))
    print("  first 5 elem_ids:", p["elem_ids"][:5])
    print("  first 5 token idx:", p["first_token_idx"][:5])

for i, (p, c) in enumerate(zip(pooling, chunks)):
    assert max(p["first_token_idx"]) < len(c["input_ids"]), f"Index overflow in chunk {i}"

print("\nPooling indices sanity check passed.")


def build_document_element_sequence(chunks, pooling):
    doc_elem_positions = OrderedDict()

    for chunk_idx, (chunk, pool) in enumerate(zip(chunks, pooling)):
        for eid, tok_idx in zip(pool["elem_ids"], pool["first_token_idx"]):
            if eid not in doc_elem_positions:
                doc_elem_positions[eid] = (chunk_idx, tok_idx)

    doc_elem_ids = list(doc_elem_positions.keys())
    return doc_elem_ids, doc_elem_positions


doc_elem_ids, doc_elem_positions = build_document_element_sequence(chunks, pooling)

ROOT_ID = 0
decoder_elem_ids = [ROOT_ID] + doc_elem_ids

print("num document elements (no root):", len(doc_elem_ids))
print("num decoder elements (with root):", len(decoder_elem_ids))

orig_elem_ids = [e["id"] for e in elements]
assert set(doc_elem_ids) == set(orig_elem_ids), "Element ID mismatch after merging chunks"

print("\nFirst 10 decoder element ids:")
print(decoder_elem_ids[:10])

sample_eid = doc_elem_ids[0]
print(
    f"\nElement {sample_eid} comes from (chunk_idx, token_idx):",
    doc_elem_positions[sample_eid],
)

print("\nCELL L sanity check passed.")