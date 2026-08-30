import time


def main():
    try:
        import torch
    except Exception as e:
        print("PyTorch import failed:", e)
        print("Hint: activate your venv and install torch")
        print("  .\\.venv\\Scripts\\Activate.ps1")
        print(
            "  pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch torchvision torchaudio"
        )
        return

    print("PyTorch version:", torch.__version__)
    print("CUDA runtime:", getattr(torch.version, "cuda", None))
    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)
    if cuda_ok:
        print("GPU count:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        bf16 = False
        if callable(is_bf16_supported):
            bf16 = bool(is_bf16_supported())
        print("bf16 supported:", bf16)

    def bench(device):
        # Use float32 on CPU, float16 on CUDA for speed
        dtype = torch.float16 if (device.type == "cuda") else torch.float32
        size = 2048
        x = torch.randn(size, size, device=device, dtype=dtype)
        y = torch.randn(size, size, device=device, dtype=dtype)
        # warmup
        for _ in range(2):
            _ = x @ y
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        z = x @ y
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0
        return dt, float(z.norm().item())

    # CPU benchmark
    import torch  # type: ignore

    cpu_dt, cpu_norm = bench(torch.device("cpu"))
    print(f"CPU matmul 2048x2048: {cpu_dt:.2f} ms (norm={cpu_norm:.2f})")

    # GPU benchmark (if available)
    if cuda_ok:
        gpu_dt, gpu_norm = bench(torch.device("cuda"))
        print(f"GPU matmul 2048x2048: {gpu_dt:.2f} ms (norm={gpu_norm:.2f})")
        speedup = cpu_dt / gpu_dt if gpu_dt > 0 else float("inf")
        print(f"Speedup vs CPU: {speedup:.1f}x")
    else:
        print("No CUDA device detected by PyTorch.")
        print("If you have an NVIDIA GPU, install the CUDA wheel:")
        print("  pip uninstall -y torch torchvision torchaudio")
        print(
            "  pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch torchvision torchaudio"
        )


if __name__ == "__main__":
    main()
