"""Train the catELMo-style MLP on a pre-encoded BLOSUM62 pkl.

Mirrors catELMo's downstream architecture:
    branch(TCR):  Linear(tcr_dim -> 2048) -> BatchNorm -> Dropout(0.3) -> SiLU
    branch(epi):  Linear(epi_dim -> 2048) -> BatchNorm -> Dropout(0.3) -> SiLU
    concat(4096) -> Linear(4096 -> 1024) -> BatchNorm -> Dropout(0.3) -> SiLU
                 -> Linear(1024 -> 1)  [logit]

Three ablation conditions with identical architecture and budget:
    full      both inputs
    tcr_only  epitope vector zeroed
    epi_only  TCR vector zeroed

PERFORMANCE NOTE. The whole encoded dataset is ~1.1 GB, so it is loaded onto
the GPU ONCE and batched by index there. With batch size 32 there are ~7,000
steps per epoch, each doing very little work; a torch DataLoader spends nearly
all of that time on Python overhead and host-to-device copies rather than on
the model. Keeping the tensors resident and slicing with randperm removes that
entirely and does not change the math.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import epitope_disjoint_split


# -- model ---------------------------------------------------------------------

class Branch(nn.Module):
    def __init__(self, in_dim, hidden=2048, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class BindingMLP(nn.Module):
    """catELMo's downstream prediction model, in PyTorch."""

    def __init__(self, tcr_dim, epi_dim, branch_hidden=2048,
                 head_hidden=1024, dropout=0.3):
        super().__init__()
        self.tcr_branch = Branch(tcr_dim, branch_hidden, dropout)
        self.epi_branch = Branch(epi_dim, branch_hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(branch_hidden * 2, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, tcr, epi):
        return self.head(
            torch.cat([self.tcr_branch(tcr), self.epi_branch(epi)], dim=1)
        ).squeeze(1)


def apply_ablation(tcr, epi, mode):
    if mode == "full":     return tcr, epi
    if mode == "tcr_only": return tcr, torch.zeros_like(epi)
    if mode == "epi_only": return torch.zeros_like(tcr), epi
    raise ValueError(f"unknown mode: {mode}")


# -- data ----------------------------------------------------------------------

def load_pkl(path):
    import pandas as pd
    df = pd.read_pickle(path)
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "tcr_embeds":          rename[c] = "tcr_embeds"
        if lc == "epi_embeds":          rename[c] = "epi_embeds"
        # data.py's splitter expects these exact names
        if lc == "binding":             rename[c] = "label"
        if lc == "partition":           rename[c] = "partition"
        if lc in ("epi", "epitope"):    rename[c] = "epitope"
        if lc in ("tcr", "cdr3"):       rename[c] = "cdr3"
    return df.rename(columns=rename)


def to_device_tensors(df, device):
    """Stack embeddings and move them onto the device once."""
    tcr = torch.from_numpy(
        np.asarray(np.stack(df["tcr_embeds"].to_numpy()), dtype=np.float32)
    ).to(device)
    epi = torch.from_numpy(
        np.asarray(np.stack(df["epi_embeds"].to_numpy()), dtype=np.float32)
    ).to(device)
    y = torch.from_numpy(
        df["label"].to_numpy(dtype=np.float32)
    ).to(device)
    return tcr, epi, y


# -- eval ----------------------------------------------------------------------

@torch.no_grad()
def predict(model, tensors, mode, batch_size):
    """Inference over GPU-resident tensors, in order."""
    model.eval()
    tcr_all, epi_all, y_all = tensors
    out = torch.empty(len(y_all), device=tcr_all.device)
    for i in range(0, len(y_all), batch_size):
        tcr, epi = apply_ablation(
            tcr_all[i:i + batch_size], epi_all[i:i + batch_size], mode
        )
        out[i:i + batch_size] = torch.sigmoid(model(tcr, epi))
    return out.cpu().numpy(), y_all.cpu().numpy()


def macro_auroc(scores, labels, epitopes, min_per_class=3):
    aucs = []
    for ep in np.unique(epitopes):
        m = epitopes == ep
        yl = labels[m]
        if yl.sum() < min_per_class or (len(yl) - yl.sum()) < min_per_class:
            continue
        aucs.append(roc_auc_score(yl, scores[m]))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


# -- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["full", "tcr_only", "epi_only"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs",     type=int,   default=200)
    ap.add_argument("--patience",   type=int,   default=30)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--eval-batch-size", type=int, default=4096,
                    help="larger batches for inference; no effect on results")
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--device",     default=None)
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 matmuls on Ampere+ (small speedup)")
    ap.add_argument("--save-preds", default=None,
                    help="write raw test predictions to this .npz for diagnosis")
    ap.add_argument("--split", default="auto",
                    choices=["auto", "partition", "epitope", "random"],
                    help="auto: use the partition column if present, else "
                         "epitope-disjoint. random: split rows at random, so "
                         "epitopes appear in BOTH train and test -- the "
                         "seen-epitope control.")
    args = ap.parse_args()

    out = args.out or f"results/pkl_{args.mode}.json"
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading {args.pkl} ...")
    df = load_pkl(args.pkl)

    mode = args.split
    if mode == "auto":
        mode = "partition" if "partition" in df.columns else "epitope"

    if mode == "partition":
        if "partition" not in df.columns:
            raise SystemExit("--split partition but the pkl has no partition column")
        # Partitions were built before the negatives (see make_swap_pkl.py).
        # Re-splitting here would mix receptors across the boundary and
        # recreate the label inversion the construction exists to prevent.
        train_df = df[df.partition == "train"].reset_index(drop=True)
        val_df = df[df.partition == "val"].reset_index(drop=True)
        test_df = df[df.partition == "test"].reset_index(drop=True)
        print("  split: precomputed partition column")
    elif mode == "random":
        # Seen-epitope control: epitopes appear in BOTH train and test, so
        # this measures what the model can learn at all, as opposed to what
        # it can transfer to epitopes it has never seen. A large gap between
        # this and the epitope-disjoint number IS the memorisation result.
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(df))
        n_test = int(0.15 * len(df))
        n_val = int(0.10 * len(df))
        test_df = df.iloc[perm[:n_test]].reset_index(drop=True)
        val_df = df.iloc[perm[n_test:n_test + n_val]].reset_index(drop=True)
        train_df = df.iloc[perm[n_test + n_val:]].reset_index(drop=True)
        shared = len(set(train_df.epitope) & set(test_df.epitope))
        print(f"  split: RANDOM rows -- {shared} epitopes appear in both "
              f"train and test (this is the point)")
    else:
        train_df, val_df, test_df = epitope_disjoint_split(df, seed=args.seed)
        print("  split: epitope-disjoint")
    print(f"mode={args.mode}  device={device}")
    print(f"train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")
    overlap = len(set(train_df.epitope) & set(test_df.epitope))
    print(f"epitopes: train {train_df.epitope.nunique()}, "
          f"val {val_df.epitope.nunique()}, "
          f"test {test_df.epitope.nunique()} "
          f"({'disjoint' if overlap == 0 else f'{overlap} SHARED with train'})\n")

    tcr_dim = len(df["tcr_embeds"].iloc[0])
    epi_dim = len(df["epi_embeds"].iloc[0])
    print(f"TCR embed dim {tcr_dim}  |  epi embed dim {epi_dim}")

    print("Moving tensors to device ...")
    tr = to_device_tensors(train_df, device)
    va = to_device_tensors(val_df, device)
    te = to_device_tensors(test_df, device)
    if device.type == "cuda":
        print(f"  GPU memory held by data: "
              f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    tcr_tr, epi_tr, y_tr = tr
    n_train = len(y_tr)

    model = BindingMLP(tcr_dim, epi_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}\n")

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state, no_improve = -1.0, None, 0
    t0 = time.time()
    epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        running = 0.0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue  # BatchNorm needs >1 sample
            tcr, epi = apply_ablation(tcr_tr[idx], epi_tr[idx], args.mode)
            y = y_tr[idx]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(tcr, epi), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)

        vs, vl = predict(model, va, args.mode, args.eval_batch_size)
        val_auc = roc_auc_score(vl, vs)
        flag = ""
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
            flag = "  <- best"
        else:
            no_improve += 1

        print(f"  epoch {epoch:3d}  loss {running/n_train:.4f}"
              f"  val AUROC {val_auc:.4f}{flag}"
              f"  patience {no_improve}/{args.patience}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

        if no_improve >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    ts, tl = predict(model, te, args.mode, args.eval_batch_size)
    test_auc = roc_auc_score(tl, ts)
    macro, n_ep = macro_auroc(ts, tl, test_df["epitope"].to_numpy())

    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds) or ".", exist_ok=True)
        np.savez(
            args.save_preds,
            scores=ts,
            labels=tl,
            epitopes=test_df["epitope"].to_numpy().astype(str),
        )
        print(f"  saved raw predictions to {args.save_preds}")

    result = {
        "mode": args.mode,
        "test_auroc": round(float(test_auc), 4),
        "macro_auroc_per_epitope": round(float(macro), 4),
        "n_epitopes_in_macro": int(n_ep),
        "best_val_auroc": round(float(best_val), 4),
        "tcr_embed_dim": tcr_dim,
        "epi_embed_dim": epi_dim,
        "n_parameters": int(n_params),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "epochs_run": epoch,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "runtime_sec": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  test AUROC              {result['test_auroc']}")
    print(f"  macro AUROC per epitope {result['macro_auroc_per_epitope']}"
          f" ({n_ep} epitopes)")
    print(f"  runtime                 {result['runtime_sec']}s")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()