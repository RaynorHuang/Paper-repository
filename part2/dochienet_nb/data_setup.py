from pathlib import Path
import json
import pandas as pd

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "dochienet_dataset"
labels_dir = DATA_DIR / "labels"
images_dir = DATA_DIR / "images"
hres_dir = DATA_DIR / "hres_images"

# splits
en_zh = json.loads((DATA_DIR / "en_zh_split.json").read_text(encoding="utf-8"))
tt = json.loads((DATA_DIR / "train_test_split.json").read_text(encoding="utf-8"))
en_ids, zh_ids = set(en_zh["en"]), set(en_zh["zh"])
train_ids, test_ids = set(tt["train"]), set(tt["test"])

def get_lang(doc_id):
    return "en" if doc_id in en_ids else ("zh" if doc_id in zh_ids else "unknown")

def get_split(doc_id):
    return "train" if doc_id in train_ids else ("test" if doc_id in test_ids else "unknown")

def list_pages(doc_id, prefer_hres=True):
    base = hres_dir if prefer_hres else images_dir
    folder = base / doc_id
    # page1.jpg, page2.jpg...
    pages = sorted(folder.glob("page*.jpg"), key=lambda p: int(p.stem.replace("page","")))
    return pages

doc_ids = sorted(list(train_ids | test_ids))
rows = []
for did in doc_ids:
    lp = labels_dir / f"{did}.json"
    pages = list_pages(did, prefer_hres=True)
    rows.append({
        "doc_id": did,
        "split": get_split(did),
        "lang": get_lang(did),
        "label_path": str(lp),
        "n_pages": len(pages),
        "page_paths": [str(p) for p in pages],
    })

index_df = pd.DataFrame(rows)
index_df.head(), index_df["split"].value_counts(), index_df["lang"].value_counts()


