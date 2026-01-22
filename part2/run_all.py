#!/usr/bin/env python3
"""
Entry-point that reproduces the typical "run the notebook top-to-bottom" flow,
while keeping expensive/optional steps guarded.

Usage:
  python run_all.py
"""
from pathlib import Path

# 1) Data index & splits
from dochienet_nb import data_setup  # builds index_df

# 2) Parsing utilities / targets
from dochienet_nb import label_parsing  # defines functions used later

# 3) Optional: demo chunking on one document (creates some demo variables)
from dochienet_nb import chunking_demo  # creates chunks/positions for a DOC_ID demo

# 4) Model(s)
from dochienet_nb import model_dhformer_mini  # defines DHFormerMini
from dochienet_nb import dataset_loader       # defines Dataset/Loader + collate_fn and builds train_loader
from dochienet_nb import model_ssa            # defines SSA model and instantiates model_ssa

# 5) Evaluation (micro P/R/F1 on test split) — expects test_loader/model_ssa
from dochienet_nb import eval_relations_f1

# 6) Optional: checkpoint eval + TEDS (only runs if checkpoint exists / deps present)
from dochienet_nb import eval_best_and_teds

if __name__ == "__main__":
    print("\nAll modules imported. If you want training, see dochienet_nb/training_cells_reference.py (these were reference cells in the notebook).")
