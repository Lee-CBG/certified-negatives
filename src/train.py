"""STEP 2 -- Train one condition and evaluate it on an epitope-disjoint test set.

Run this three times, once per ablation mode:

    full      both inputs
    tcr_only  epitope zeroed
    epi_only  CDR3 zeroed

Everything else is held identical: architecture, parameter count, seed, epochs,
batch size, learning rate, and the exact train/val/test partition. The only
thing that varies is which input is available.

Reported metrics:
    test AUROC        -- pooled over all test pairs
    macro AUROC       -- AUROC computed per epitope, then averaged. This is the
                         honest one: it cannot be inflated by getting the
                         easy/frequent epitopes right.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import (
    encode_partition,
    epitope_disjoint_split,
    load_pairs,
    subsample,
)
from model import BindingCNN, apply_ablation


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(epi, tcr, y, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(epi), torch.from_numpy(tcr), torch.from_numpy(y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader, mode, device):
    model.eval()
    scores, labels = [], []
    for epi, tcr, y in loader:
        epi, tcr = epi.to(device), tcr.to(device)
        epi, tcr = apply_ablation(epi, tcr, mode)
        scores.append(torch.sigmoid(model(epi, tcr)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def macro_auroc(scores, labels, epitopes, min_per_class=3):
    """AUROC per epitope, averaged. Epitopes without both classes are skipped."""
    aucs = []
    for ep in np.unique(epitopes):
        m = epitopes == ep
        yl = labels[m]
        if yl.sum() < min_per_class or (len(yl) - yl.sum()) < min_per_class:
            continue
        aucs.append(roc_auc_score(yl, scores[m]))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--mode", required=True, choices=["full", "tcr_only", "epi_only"]
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-rows", type=int, default=120_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-filters", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = args.out or f"results/{args.mode}.json"
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    # Identical data for every mode: same seed, same subsample, same split.
    df = load_pairs(args.csv)
    df = subsample(df, args.max_rows, seed=args.seed)
    train_df, val_df, test_df = epitope_disjoint_split(df, seed=args.seed)

    print(f"mode={args.mode}  device={device}")
    print(
        f"train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,} pairs"
    )
    print(
        f"epitopes: train {train_df.epitope.nunique()}, "
        f"val {val_df.epitope.nunique()}, test {test_df.epitope.nunique()} (disjoint)\n"
    )

    tr = make_loader(*encode_partition(train_df), args.batch_size, True)
    va = make_loader(*encode_partition(val_df), args.batch_size, False)
    te = make_loader(*encode_partition(test_df), args.batch_size, False)

    model = BindingCNN(args.n_filters, args.hidden, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} (identical across all three modes)\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for epi, tcr, y in tr:
            epi, tcr, y = epi.to(device), tcr.to(device), y.to(device)
            epi, tcr = apply_ablation(epi, tcr, args.mode)
            opt.zero_grad()
            loss = loss_fn(model(epi, tcr), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)

        vs, vl = predict(model, va, args.mode, device)
        val_auc = roc_auc_score(vl, vs)
        flag = ""
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(
            f"  epoch {epoch:2d}  loss {running/len(train_df):.4f}  "
            f"val AUROC {val_auc:.4f}{flag}"
        )

    model.load_state_dict(best_state)
    ts, tl = predict(model, te, args.mode, device)
    test_auc = roc_auc_score(tl, ts)
    macro, n_ep = macro_auroc(ts, tl, test_df["epitope"].to_numpy())

    result = {
        "mode": args.mode,
        "test_auroc": round(float(test_auc), 4),
        "macro_auroc_per_epitope": round(float(macro), 4),
        "n_epitopes_in_macro": int(n_ep),
        "best_val_auroc": round(float(best_val), 4),
        "n_parameters": int(n_params),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "epochs": args.epochs,
        "seed": args.seed,
        "runtime_sec": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  test AUROC              {result['test_auroc']}")
    print(f"  macro AUROC per epitope {result['macro_auroc_per_epitope']} "
          f"({n_ep} epitopes)")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()