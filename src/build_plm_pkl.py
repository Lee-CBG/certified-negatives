#!/usr/bin/env python3
"""Rebuild existing pkls with protein-language-model embeddings in place of BLOSUM62.

Partitions, labels and negative identities are taken unchanged from the source
pkls, so every output is paired cell-for-cell with the BLOSUM runs and only the
representation differs.

    # validate the whole pipeline in seconds, no GPU, no weights
    python3 build_plm_pkl.py --dry-run \
        --src data/m4g_seed0.pkl --tag m4g --out-dir data

    # real run: embeds once, writes every requested construction
    python3 build_plm_pkl.py \
        --model facebook/esm2_t33_650M_UR50D --short esm650 \
        --src data/m4g_seed0.pkl --src data/m4g_seed1.pkl --src data/m4g_seed2.pkl \
        --src data/grouped_seed0.pkl --src data/grouped_seed1.pkl --src data/grouped_seed2.pkl \
        --out-dir data --cache data/plm_cache_esm650.npz

Other models, same code path:
    --model wukevin/tcr-bert            --short tcrbert  --space-residues
    --model facebook/esm2_t30_150M_UR50D --short esm150
    --model Rostlab/prot_bert           --short protbert --space-residues
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd


def unique_seqs(paths):
    tcr, epi = set(), set()
    meta = {}
    for p in paths:
        df = pd.read_pickle(p)
        for c in ("epi", "tcr", "binding", "partition"):
            if c not in df.columns:
                sys.exit(f"{p}: missing column {c} (has {list(df.columns)})")
        tcr |= set(df["tcr"].astype(str))
        epi |= set(df["epi"].astype(str))
        meta[p] = len(df)
        del df
    return sorted(tcr), sorted(epi), meta


def embed(seqs, model_name, space, batch_tokens, device, dry):
    if dry:
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(seqs), 32), dtype=np.float32) * 0.1

    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval().half()
    dim = mdl.config.hidden_size
    out = np.zeros((len(seqs), dim), dtype=np.float16)

    order = np.argsort([len(s) for s in seqs])          # length-bucketed batching
    i = 0
    with torch.no_grad():
        while i < len(order):
            L = len(seqs[order[i]]) + 2
            n = max(1, batch_tokens // max(L, 1))
            idx = order[i:i + n]
            batch = [" ".join(seqs[j]) if space else seqs[j] for j in idx]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=True).to(device)
            h = mdl(**enc).last_hidden_state                # B x T x D
            m = enc["attention_mask"].clone()
            m[:, 0] = 0                                     # drop BOS
            lens = enc["attention_mask"].sum(1) - 1
            m[torch.arange(m.size(0), device=m.device), lens] = 0   # drop EOS
            mm = m.unsqueeze(-1).to(h.dtype)
            pooled = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            out[idx] = pooled.float().cpu().numpy().astype(np.float16)
            i += n
            if (i // max(n, 1)) % 50 == 0:
                print(f"    {i:,}/{len(order):,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="source pkl; repeat. Output name reuses its basename.")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--short", default=None, help="prefix for output pkls")
    ap.add_argument("--tag", default=None, help="alias for --short")
    ap.add_argument("--space-residues", action="store_true",
                    help="insert spaces between residues (TCR-BERT, ProtBERT)")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default=None, help="npz embedding cache to reuse")
    ap.add_argument("--dry-run", action="store_true",
                    help="32-dim random embeddings, no model load, no GPU")
    a = ap.parse_args()
    short = a.short or a.tag or re.sub(r"[^a-z0-9]", "", a.model.split("/")[-1].lower())[:10]

    tcr, epi, meta = unique_seqs(a.src)
    print(f"{len(a.src)} source pkls, {sum(meta.values()):,} rows total")
    print(f"unique receptors {len(tcr):,}   unique epitopes {len(epi):,}")
    print(f"embedding {len(tcr)+len(epi):,} sequences once, not "
          f"{2*sum(meta.values()):,} row vectors")

    if a.cache and os.path.exists(a.cache) and not a.dry_run:
        z = np.load(a.cache, allow_pickle=True)
        if list(z["tcr_keys"]) == tcr and list(z["epi_keys"]) == epi:
            print(f"reusing cache {a.cache}")
            TE, EE = z["tcr_emb"], z["epi_emb"]
        else:
            sys.exit(f"{a.cache} exists but its key set differs; delete or rename it")
    else:
        print(f"model={a.model}  space_residues={a.space_residues}  dry={a.dry_run}")
        print("  receptors ...", flush=True)
        TE = embed(tcr, a.model, a.space_residues, a.batch_tokens, a.device, a.dry_run)
        print("  epitopes ...", flush=True)
        EE = embed(epi, a.model, a.space_residues, a.batch_tokens, a.device, a.dry_run)
        if a.cache and not a.dry_run:
            np.savez(a.cache, tcr_keys=np.array(tcr), epi_keys=np.array(epi),
                     tcr_emb=TE, epi_emb=EE)
            print(f"wrote cache {a.cache}")
    print(f"receptor dim {TE.shape[1]}  epitope dim {EE.shape[1]}")

    ti = {s: k for k, s in enumerate(tcr)}
    ei = {s: k for k, s in enumerate(epi)}
    for p in a.src:
        df = pd.read_pickle(p)
        base = os.path.basename(p).replace(".pkl", "")
        out = pd.DataFrame({
            "epi": df["epi"].to_numpy(),
            "tcr": df["tcr"].to_numpy(),
            "binding": df["binding"].to_numpy().astype(int),
            "partition": df["partition"].to_numpy(),
        })
        out["tcr_embeds"] = list(TE[[ti[s] for s in df["tcr"].astype(str)]])
        out["epi_embeds"] = list(EE[[ei[s] for s in df["epi"].astype(str)]])
        dst = os.path.join(a.out_dir, f"{short}_{base}.pkl")
        out.to_pickle(dst)
        pr = out["binding"].mean()
        print(f"wrote {dst}  rows={len(out):,}  pos={pr:.4f}  "
              f"parts={out['partition'].value_counts().to_dict()}")
        del df, out


if __name__ == "__main__":
    main()