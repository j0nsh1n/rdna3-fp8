"""FP8 weights in VRAM for PyTorch on gfx1100: a linear layer backed by the decode prototype (proto/gemv/fp8_gemv.hip).

Build the library first:
  hipcc -O3 -std=c++17 --offload-arch=gfx1100 -fPIC -shared proto/torch/fp8_ext.hip -o build/torch-ext/libfp8ext.so \
      -L/usr/lib64 -lamdhip64
Run the checks and timings (device 0 must be the RX 7900 XT):
  build/torch-venv/bin/python -I proto/torch/fp8_linear.py
Variant a keeps F32 activations (exact, like llama.cpp); variant b rounds activations to FP16 (faster at batch >= 4).
Batch sizes 1..8 only (decode). The input is converted to F32 and the output back to the input dtype; both count in
the timings.
"""
import ctypes
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch  # noqa: E402  (must load PyTorch's HIP runtime before ours so both share it)
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(str(ROOT / "build/torch-ext/libfp8ext.so"))
LIB.fp8_gemv_run.restype = ctypes.c_int
LIB.fp8_gemv_run.argtypes = [ctypes.c_char, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 5 + [
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]

# Best rows per wave from notes/prototype-results-1.md.
BEST_R = {"a": {1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 4, 8: 2},
          "b": {1: 2, 2: 2, 3: 1, 4: 2, 5: 2, 6: 1, 7: 4, 8: 4}}


class FP8Linear:
    def __init__(self, weight_fp8: torch.Tensor, scale: float, variant: str = "a"):
        assert weight_fp8.dtype == torch.float8_e4m3fn and weight_fp8.is_cuda and weight_fp8.dim() == 2
        assert weight_fp8.shape[1] % 16 == 0 and variant in BEST_R
        self.w = weight_fp8.contiguous().view(torch.uint8)
        self.out_features, self.in_features = self.w.shape
        self.scale = float(scale)
        self.variant = variant
        dev = self.w.device
        self.yh = torch.empty(8 * self.in_features // 2, dtype=torch.int32, device=dev)
        self.ovf = torch.empty(1, dtype=torch.int32, device=dev)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        assert x.dim() == 2 and x.shape[1] == self.in_features and 1 <= n <= 8
        xf = x.float().contiguous()
        out = torch.empty(n, self.out_features, dtype=torch.float32, device=x.device)
        rc = LIB.fp8_gemv_run(self.variant.encode(), BEST_R[self.variant][n], n, self.w.data_ptr(), xf.data_ptr(),
                              self.yh.data_ptr(), self.ovf.data_ptr(), out.data_ptr(), self.out_features,
                              self.in_features, self.scale * 256.0, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"fp8_gemv_run failed with HIP error {rc}")
        return out.to(x.dtype)


def hip_runtimes_loaded():
    return sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libamdhip64" in ln})


def time_us(fn, warm=20, iters=200):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000.0 for s, e in ev)
    return statistics.median(t), t[0], t[-1]


def main():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1100"):
        sys.exit(f"device 0 is {arch}, not gfx1100")
    runtimes = hip_runtimes_loaded()
    print("HIP runtimes mapped:", runtimes)
    if len(runtimes) != 1:
        sys.exit("more than one HIP runtime in the process")

    M, K, scale = 16384, 14336, 0.0123
    torch.manual_seed(0)
    w8 = (torch.randn(M, K, device="cuda") * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    w16 = w8.to(torch.float16) * scale
    layers = {v: FP8Linear(w8, scale, v) for v in "ab"}

    # Correctness on the first 64 rows against float64.
    wref = (w8[:64].to(torch.float64) * scale).cpu()
    for n in (1, 4, 8):
        x = torch.randn(n, K, device="cuda", dtype=torch.float16)
        ref = (x.to(torch.float64).cpu() @ wref.T)
        for v, layer in layers.items():
            got = layer(x).to(torch.float64).cpu()[:, :64]
            rel = ((got - ref).abs().max() / ref.abs().max()).item()
            print(f"check variant {v} n={n}: max |err| / max |ref| = {rel:.2e}")
            if not rel < 2e-3:
                sys.exit("correctness check failed")

    # NaN code in row 5 must give NaN in row 5 only.
    wn = w8.clone()
    wn.view(torch.uint8)[5, 77] = 0x7F
    x = torch.randn(4, K, device="cuda", dtype=torch.float16)
    for v in "ab":
        out = FP8Linear(wn, scale, v)(x)
        nan_rows = torch.isnan(out).any(dim=0).nonzero().flatten().tolist()
        print(f"nan check variant {v}: rows with NaN {nan_rows}, all columns NaN in row 5: {bool(torch.isnan(out[:, 5]).all())}")
        if nan_rows != [5] or not torch.isnan(out[:, 5]).all():
            sys.exit("NaN check failed")
    del wn

    print(f"{'n':>2} {'F16 F.linear us':>16} {'FP8 a us':>10} {'FP8 b us':>10}  (median; min-max)")
    for n in range(1, 9):
        x = torch.randn(n, K, device="cuda", dtype=torch.float16)
        f = time_us(lambda: F.linear(x, w16))
        a = time_us(lambda: layers["a"](x))
        b = time_us(lambda: layers["b"](x))
        print(f"{n:2d} {f[0]:9.1f} ({f[1]:.0f}-{f[2]:.0f}) {a[0]:8.1f} ({a[1]:.0f}-{a[2]:.0f}) "
              f"{b[0]:8.1f} ({b[1]:.0f}-{b[2]:.0f})")


if __name__ == "__main__":
    main()
