# RE0_clean.ipynb split into importable Python modules

This folder was generated from the notebook `RE0_clean.ipynb` and keeps the same logic while making the code importable.

## Structure

- `re0/`
  - `dataset.py`          Dataset + label mapping + meta filtering
  - `priors.py`           Prior estimation helpers
  - `config.py`           CFG + seeding + common helpers (collate, box normalization, etc.)
  - `embedders.py`        Text/layout/visual embedders
  - `model.py`            DSPSModel
  - `losses.py`           FocalLoss + loss aggregation
  - `train.py`            Training loop(s)
  - `predict.py`          Inference + export utilities
  - `metrics.py`          STEDS / TED evaluation implementation
  - `reading_order.py`    Text normalization + reading order helpers
  - `experiments.py`      Diagnostic / experiment wrappers

- `main.py` reproduces the notebook's training + export steps.

## Run

Edit dataset paths in `main.py` or set env vars:

- `HRDH_ROOT`
- `HRDS_ROOT` (only needed for some experiment helpers)
- `RUN_ROOT`  (optional)

Then:

```bash
python main.py
```

