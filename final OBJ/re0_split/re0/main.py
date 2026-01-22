"""Entry point that reproduces the original notebook execution order.

Notes:
- The original notebook contains Windows-local paths for HRDH/HRDS datasets.
  Update HRDH_ROOT/HRDS_ROOT below (or pass via environment variables) before running.
"""

import os
import torch

from re0.config import CFG, seed_everything, collate_doc
from re0.dataset import HRDHDataset, LABEL2ID_14, REL3
from re0.priors import compute_M_cp_from_dataset
from re0.model import DSPSModel
from re0.losses import FocalLoss
from re0.train import train_one_epoch
from re0.predict import export_split_predictions


def main() -> None:
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== dataset roots (edit as needed) =====
    HRDH_ROOT = os.environ.get("HRDH_ROOT", r"C:\Users\tomra\Desktop\PAPER\final OBJ\HRDH")
    HRDS_ROOT = os.environ.get("HRDS_ROOT", r"C:\Users\tomra\Desktop\PAPER\final OBJ\HRDS")
    RUN_ROOT  = os.environ.get("RUN_ROOT", "hrds_runs")

    # ===== config =====
    cfg = CFG()
    # The notebook overrides:
    cfg.epochs = 2
    cfg.num_workers = 0

    # ===== dataset =====
    train_ds = HRDHDataset(HRDH_ROOT, split="train", cfg=cfg)
    test_ds  = HRDHDataset(HRDH_ROOT, split="test",  cfg=cfg)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_doc
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_doc
    )

    # ===== priors / sizes =====
    num_classes = len(LABEL2ID_14)
    num_rel = len(REL3)
    M_cp = compute_M_cp_from_dataset(train_ds, num_classes=num_classes)

    # ===== model =====
    model = DSPSModel(
        num_classes=num_classes,
        num_rel=num_rel,
        M_cp=M_cp,
        cfg=cfg,
        use_text=False,
        use_visual=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    focal_cls = FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha).to(device)
    focal_rel = FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha).to(device)

    best = 1e9
    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, cfg, focal_cls, focal_rel)
        print(ep, tr)
        if tr.get("loss", 1e9) < best:
            best = tr["loss"]
            torch.save(model.state_dict(), "dsps_hrdh_best.pt")
            print("saved: dsps_hrdh_best.pt")

    # ===== export predictions (notebook step) =====
    if os.path.exists("dsps_hrdh_best.pt"):
        model.load_state_dict(torch.load("dsps_hrdh_best.pt", map_location=device))
    export_split_predictions(model, test_loader, save_dir="exports_hrdh_test", device=device)


if __name__ == "__main__":
    main()
