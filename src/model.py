"""NetTCR-style 1D CNN for TCR-epitope binding prediction.

Two branches (epitope, CDR3), each with parallel convolutions at kernel sizes
1/3/5/7/9, global max-pooling, concatenation, then a dense head.

The ablation is applied by ZEROING AN INPUT, not by deleting a branch. That
keeps parameter count, optimiser state and training budget identical across all
three conditions, so the comparison is clean: any AUROC difference is
attributable to the information in the input, not to a smaller model.
"""

import torch
import torch.nn as nn

KERNELS = (1, 3, 5, 7, 9)


class ConvBranch(nn.Module):
    """Parallel convolutions at several kernel widths, then global max-pool."""

    def __init__(self, in_channels=20, n_filters=32):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(in_channels, n_filters, kernel_size=k, padding=k // 2)
            for k in KERNELS
        )
        self.act = nn.SiLU()
        self.out_dim = n_filters * len(KERNELS)

    def forward(self, x):
        # x: (batch, length, 20) -> conv1d wants (batch, 20, length)
        x = x.transpose(1, 2)
        pooled = [self.act(conv(x)).max(dim=2).values for conv in self.convs]
        return torch.cat(pooled, dim=1)


class BindingCNN(nn.Module):
    def __init__(self, n_filters=32, hidden=64, dropout=0.3):
        super().__init__()
        self.epi_branch = ConvBranch(n_filters=n_filters)
        self.tcr_branch = ConvBranch(n_filters=n_filters)
        concat_dim = self.epi_branch.out_dim + self.tcr_branch.out_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(concat_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, epi, tcr):
        h = torch.cat([self.epi_branch(epi), self.tcr_branch(tcr)], dim=1)
        return self.head(h).squeeze(1)


def apply_ablation(epi, tcr, mode: str):
    """Zero out one input tensor.

    full     -- both inputs intact (the baseline)
    tcr_only -- epitope zeroed; can the model predict from the CDR3 alone?
    epi_only -- CDR3 zeroed; can the model predict from the peptide alone?
    """
    if mode == "full":
        return epi, tcr
    if mode == "tcr_only":
        return torch.zeros_like(epi), tcr
    if mode == "epi_only":
        return epi, torch.zeros_like(tcr)
    raise ValueError(f"unknown ablation mode: {mode}")