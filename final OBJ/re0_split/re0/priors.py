from __future__ import annotations
from typing import Any, Dict, List

import numpy as np

from .dataset import HRDHDataset


def compute_M_cp_from_dataset(dataset: HRDHDataset, num_classes: int, pseudo_count: float = 5.0):
    """
    返回 M_cp: shape (num_classes+1, num_classes)
    行：parent_class + ROOT
    列：child_class
    """
    ROOT = num_classes  # extra row index
    counts = np.zeros((num_classes + 1, num_classes), dtype=np.float64)

    for i in range(len(dataset)):
        doc = dataset[i]
        y_cls = doc["y_cls"]
        y_parent = doc["y_parent"]
        L = len(y_cls)

        for child in range(L):
            c = y_cls[child]
            p = y_parent[child]
            if p == -1:
                pc = ROOT
            else:
                pc = y_cls[p]
            counts[pc, c] += 1.0

    # additive smoothing per column
    counts += pseudo_count
    # normalize columns -> probability
    col_sum = counts.sum(axis=0, keepdims=True)
    M_cp = counts / np.clip(col_sum, 1e-12, None)
    return M_cp

def compute_M_cp_from_dataset(dataset, num_classes: int, pseudo_count: float = 5.0):
    """
    M_cp shape: (num_classes+1, num_classes)
      rows: parent_class + ROOT(row = num_classes)
      cols: child_class
    Column-normalized: sum over rows for each col = 1
    """
    ROOT = num_classes
    counts = np.zeros((num_classes + 1, num_classes), dtype=np.float64)

    for di in range(len(dataset)):
        doc = dataset[di]
        y_cls = doc["y_cls"]
        y_parent = doc["y_parent"]
        L = len(y_cls)

        for child in range(L):
            c = int(y_cls[child])
            p = int(y_parent[child])
            if p == -1:
                pc = ROOT
            else:
                pc = int(y_cls[p])
            counts[pc, c] += 1.0

    # Additive smoothing
    counts += float(pseudo_count)

    # Column normalize
    col_sum = counts.sum(axis=0, keepdims=True)
    M_cp = counts / np.clip(col_sum, 1e-12, None)
    return M_cp, counts

def top_parent_for_each_child(M_cp: np.ndarray, id2label: list, topk: int = 6):
    """
    For each child class, list top-k parent classes (including ROOT).
    """
    num_classes = len(id2label)
    ROOT = num_classes

    rows, cols = M_cp.shape
    assert rows == num_classes + 1 and cols == num_classes

    for child in range(num_classes):
        probs = M_cp[:, child]
        idxs = np.argsort(-probs)[:topk]
        child_name = id2label[child]
        print(f"\n[Child] {child} {child_name}")
        for r in idxs:
            parent_name = "ROOT" if r == ROOT else id2label[r]
            print(f"  parent={parent_name:<12s}  P={probs[r]:.4f}")

def top_edges_global(counts: np.ndarray, id2label: list, topn: int = 30):
    """
    Print global most frequent (parent, child) pairs from raw counts (before normalization).
    """
    num_classes = len(id2label)
    ROOT = num_classes

    flat = []
    for pc in range(num_classes + 1):
        for cc in range(num_classes):
            flat.append((counts[pc, cc], pc, cc))
    flat.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop {topn} (parent, child) by COUNT (after smoothing included):")
    for k in range(topn):
        cnt, pc, cc = flat[k]
        parent_name = "ROOT" if pc == ROOT else id2label[pc]
        child_name = id2label[cc]
        print(f"{k:02d}  {parent_name:<12s} -> {child_name:<12s}  count={cnt:.1f}")
