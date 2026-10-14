"""Verify the GPU is usable by THIS model, not just that torch imports.

`torch.cuda.is_available()` returning True is necessary but not sufficient --
a driver/wheel mismatch can pass that check and then fail on the first kernel
launch. So this runs a real forward and backward pass through the actual
BindingCNN on GPU-resident tensors, which is the thing that has to work.
"""

import sys

import torch

sys.path.insert(0, "src")


def main():
    print(f"torch version          {torch.__version__}")
    print(f"built for CUDA         {torch.version.cuda or 'CPU-only build'}")
    print(f"cuda.is_available()    {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print()
        print("No GPU visible to torch. Either:")
        print("  - you installed the CPU wheel  -> ./setup_env.sh cu124")
        print("  - the driver is older than the wheel's CUDA version")
        print("    (check the CUDA Version field in nvidia-smi and pick a")
        print("     lower tag, e.g. ./setup_env.sh cu121)")
        print()
        print("The code still runs on CPU, just slower (~25 min).")
        return 1

    n = torch.cuda.device_count()
    print(f"device count           {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(
            f"  [{i}] {p.name}, {p.total_memory / 1024**3:.1f} GB, "
            f"capability {p.major}.{p.minor}"
        )

    print("\nRunning a real forward+backward through BindingCNN on GPU...")
    from data import MAX_EPI_LEN, MAX_TCR_LEN
    from model import BindingCNN, apply_ablation

    dev = torch.device("cuda")
    model = BindingCNN().to(dev)
    epi = torch.randn(64, MAX_EPI_LEN, 20, device=dev)
    tcr = torch.randn(64, MAX_TCR_LEN, 20, device=dev)
    y = torch.randint(0, 2, (64,), device=dev).float()

    for mode in ("full", "tcr_only", "epi_only"):
        e, t = apply_ablation(epi, tcr, mode)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(e, t), y)
        loss.backward()
        print(f"  {mode:9s} ok, loss {loss.item():.4f}")
    torch.cuda.synchronize()

    used = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\npeak memory for a 64-pair batch: {used:.1f} MB")
    print("\nGPU is usable. Run ./run_all.sh -- it will pick CUDA automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())