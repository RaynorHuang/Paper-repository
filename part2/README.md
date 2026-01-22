# Split from `2.ipynb`

This folder contains a minimal, **behavior-preserving** split of the original Jupyter notebook into importable `.py` modules.

## Layout

- `dochienet_nb/data_setup.py`  
  Dataset paths, splits, and `index_df`.

- `dochienet_nb/label_parsing.py`  
  Label parsing, parent-map building, box normalization, and related helpers.

- `dochienet_nb/chunking_demo.py`  
  The notebook's demo chunking/packing cells (creates demo variables such as `chunks`, etc.).

- `dochienet_nb/dataset_loader.py`  
  `DocHieNetDocDataset`, `collate_fn`, and an example `train_loader`.

- `dochienet_nb/model_dhformer_mini.py`  
  `DHFormerMini` model wrapper.

- `dochienet_nb/model_ssa.py`  
  SSA decoder + `model_ssa` instantiation.

- `dochienet_nb/eval_relations_f1.py`  
  Relation-level micro P/R/F1 evaluation.

- `dochienet_nb/eval_best_and_teds.py`  
  Best-checkpoint loading (if available) and TEDS computations.

- `dochienet_nb/training_cells_reference.py`  
  Training cells as **non-executing reference** (they were triple-quoted in the notebook).

- `run_all.py`  
  Imports modules in notebook-like order to reproduce the same state and prints.

## Run

```bash
python run_all.py
```

If the notebook expects local files (dataset folders, checkpoints), ensure you have the same relative paths as in the notebook (`dochienet_dataset/...`).

## Notes on equivalence

- The notebook contained multiple training cells wrapped in triple-quotes; those blocks were not executed in the notebook by default.  
  They remain reference code in `training_cells_reference.py`.
- Checkpoint loading in `eval_best_and_teds.py` will fail if `best_model_ssa_fast.pt` is missing; add the file or guard the call.
