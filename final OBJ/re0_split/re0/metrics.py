from __future__ import annotations
import os, json, glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

def load_pair_files(export_dir: str) -> List[Tuple[str, str]]:
    gt_files = sorted(glob.glob(os.path.join(export_dir, "*.gt.json")))
    pairs = []
    for gt in gt_files:
        pred = gt.replace(".gt.json", ".pred.json")
        if os.path.exists(pred):
            pairs.append((gt, pred))
    return pairs

pairs = load_pair_files("exports_hrdh_test")


from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class TNode:
    label: int
    text: str
    oid: int                 # original id，用来排序
    children: List["TNode"]

    def __init__(self, label: int, text: str, oid: int):
        self.label = label
        self.text = text
        self.oid = oid
        self.children = []

def find_effective_parent(i: int) -> int:
    """
    返回 i 的有效父节点：
    - 若 parent 指向越界/非法：视为无父节点（-1），挂到 ROOT
    - 若 parent 指向被排除节点（meta 或不在 kept 集）：沿 parent 链向上跳
    - 增加循环保护，避免死循环
    """
    n = len(parent)

    def _is_valid_idx(x: int) -> bool:
        return isinstance(x, int) and (0 <= x < n)

    p = parent[i] if 0 <= i < n else -1

    # 1) 直接非法或 root
    if (p is None) or (not isinstance(p, int)) or (p < 0):
        return -1
    if not _is_valid_idx(p):
        return -1

    # 2) 向上跳，直到遇到 kept 的父节点或 root/非法
    seen = set()
    while True:
        if p in kept_set:
            return p

        if p in seen:
            # 出现环，按无父节点处理
            return -1
        seen.add(p)

        # 向上跳
        pp = parent[p]
        if (pp is None) or (not isinstance(pp, int)) or (pp < 0):
            return -1
        if not _is_valid_idx(pp):
            return -1
        p = pp

    # 找“跳过 meta 后的最近祖先”
    def find_effective_parent(i: int) -> int:
        p = parent[i]
        seen = set()
        while True:
            if p < 0:
                return -1
            if p in seen:         # 防环保护
                return -1
            seen.add(p)
            if kept(p):
                return p
            # p 是 meta 或被排除节点，继续向上跳
            p = parent[p]

    # 连接
    for i in kept_ids:
        ep = find_effective_parent(i)
        if ep < 0:
            root.children.append(obj[i])
        else:
            obj[ep].children.append(obj[i])

    # children 按原始 reading-order id 排序，确保“有序树”一致
    def sort_rec(x: TNode):
        x.children.sort(key=lambda c: c.oid)
        for ch in x.children:
            sort_rec(ch)

    sort_rec(root)
    return root, len(kept_ids)



def ted_zhang_shasha(t1: TNode, t2: TNode, match_mode: str = "strict") -> int:
    """
    Ordered Tree Edit Distance (Zhang-Shasha)
    Costs: ins=1, del=1
    match_mode:
      - "strict": label + text 完全一致 cost=0，否则 1
      - "label": 只要 label 一致 cost=0，否则 1
    """
    def postorder(root: TNode):
        nodes = []
        def dfs(x: TNode):
            for ch in x.children:
                dfs(ch)
            nodes.append(x)
        dfs(root)
        return nodes

    A = postorder(t1)
    B = postorder(t2)

    idxA = {id(node): i+1 for i, node in enumerate(A)}
    idxB = {id(node): i+1 for i, node in enumerate(B)}

    n, m = len(A), len(B)

    def leftmost_indices(nodes, idx_map):
        l = [0] * (len(nodes) + 1)  # 1-based
        for i, node in enumerate(nodes, start=1):
            cur = node
            while cur.children:
                cur = cur.children[0]
            l[i] = idx_map[id(cur)]
        return l

    l1 = leftmost_indices(A, idxA)
    l2 = leftmost_indices(B, idxB)

    def keyroots(leftmost):
        seen = {}
        for i in range(1, len(leftmost)):
            seen[leftmost[i]] = i
        return sorted(seen.values())

    kr1 = keyroots(l1)
    kr2 = keyroots(l2)

    treedist = [[0] * (m + 1) for _ in range(n + 1)]

    def relabel_cost(i: int, j: int) -> int:
        a = A[i-1]
        b = B[j-1]
        if match_mode == "label":
            return 0 if (a.label == b.label) else 1
        return 0 if (a.label == b.label and a.text == b.text) else 1

    for i in kr1:
        for j in kr2:
            i0 = l1[i]
            j0 = l2[j]
            fd = [[0] * (j - j0 + 2) for _ in range(i - i0 + 2)]

            for di in range(1, i - i0 + 2):
                fd[di][0] = fd[di-1][0] + 1  # delete
            for dj in range(1, j - j0 + 2):
                fd[0][dj] = fd[0][dj-1] + 1  # insert

            for di in range(1, i - i0 + 2):
                for dj in range(1, j - j0 + 2):
                    ii = i0 + di - 1
                    jj = j0 + dj - 1
                    if l1[ii] == i0 and l2[jj] == j0:
                        fd[di][dj] = min(
                            fd[di-1][dj] + 1,
                            fd[di][dj-1] + 1,
                            fd[di-1][dj-1] + relabel_cost(ii, jj)
                        )
                        treedist[ii][jj] = fd[di][dj]
                    else:
                        fd[di][dj] = min(
                            fd[di-1][dj] + 1,
                            fd[di][dj-1] + 1,
                            fd[di-1][dj-1] + treedist[ii][jj]
                        )

    return treedist[n][m]

def compute_steds(gt_js: Dict[str,Any], pr_js: Dict[str,Any], exclude_meta: bool = True, match_mode: str = "strict") -> float:
    gt_root, gt_n = build_tree_from_export(gt_js, exclude_meta=exclude_meta)
    pr_root, pr_n = build_tree_from_export(pr_js, exclude_meta=exclude_meta)
    dist = ted_zhang_shasha(gt_root, pr_root, match_mode=match_mode)
    denom = max(gt_n, pr_n, 1)
    return 1.0 - dist / denom

def compute_aux_metrics(gt_js: Dict[str,Any], pr_js: Dict[str,Any], exclude_meta: bool = True) -> Dict[str,float]:
    gt_nodes = gt_js["nodes"]
    pr_nodes = pr_js["nodes"]
    L = min(len(gt_nodes), len(pr_nodes))

    mask = []
    for i in range(L):
        is_meta = bool(gt_nodes[i].get("is_meta", False))
        mask.append((not is_meta) if exclude_meta else True)

    def acc(key: str) -> float:
        tot = 0
        hit = 0
        for i in range(L):
            if not mask[i]:
                continue
            tot += 1
            hit += int(gt_nodes[i][key] == pr_nodes[i][key])
        return hit / tot if tot else 0.0

    # label_id / parent / rel_id
    return {
        "cls_acc": acc("label_id"),
        "parent_acc": acc("parent"),
        "rel_acc": acc("rel_id"),
        "n_eval": float(sum(mask)),
    }

import numpy as np
from tqdm import tqdm

def eval_export_dir(export_dir: str, exclude_meta: bool = True):
    pairs = load_pair_files(export_dir)
    steds_list = []
    aux_list = []

    for gt_path, pr_path in tqdm(pairs, desc=f"eval {export_dir}"):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_js = json.load(f)
        with open(pr_path, "r", encoding="utf-8") as f:
            pr_js = json.load(f)

        st = compute_steds(gt_js, pr_js, exclude_meta=exclude_meta)
        aux = compute_aux_metrics(gt_js, pr_js, exclude_meta=exclude_meta)

        steds_list.append(st)
        aux_list.append(aux)

    steds_arr = np.array(steds_list, dtype=float)
    cls_acc = np.array([a["cls_acc"] for a in aux_list], dtype=float)
    par_acc = np.array([a["parent_acc"] for a in aux_list], dtype=float)
    rel_acc = np.array([a["rel_acc"] for a in aux_list], dtype=float)

    report = {
        "num_docs": len(pairs),
        "exclude_meta": exclude_meta,
        "STEDS_mean": float(steds_arr.mean()) if len(steds_arr) else 0.0,
        "STEDS_std": float(steds_arr.std()) if len(steds_arr) else 0.0,
        "STEDS_median": float(np.median(steds_arr)) if len(steds_arr) else 0.0,
        "cls_acc_mean": float(cls_acc.mean()) if len(cls_acc) else 0.0,
        "parent_acc_mean": float(par_acc.mean()) if len(par_acc) else 0.0,
        "rel_acc_mean": float(rel_acc.mean()) if len(rel_acc) else 0.0,
    }
    return report, steds_list, aux_list
