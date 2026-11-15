import os, sys, traceback
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
import torch.nn.functional as F
arch = torch.cuda.get_device_properties(0).gcnArchName
print("torch", torch.__version__, "arch", arch)
if not arch.startswith("gfx1100"):
    sys.exit("not gfx1100")
M, K = 16384, 14336
def attempt(name, fn):
    try:
        r = fn(); torch.cuda.synchronize()
        print(f"OK    {name}: {getattr(r,'dtype',None)} {tuple(getattr(r,'shape',()))}")
    except Exception as e:
        print(f"RAISE {name}: {type(e).__name__}: {e}")
for dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
    print("=====", dt)
    attempt("create .to(fp8) from f16", lambda: torch.randn(M, K, device="cuda", dtype=torch.float16).to(dt))
    w = torch.randn(M, K, device="cuda", dtype=torch.float16).to(dt)
    sc = torch.tensor(1.0, device="cuda")
    for n in (1, 8):
        x8 = torch.randn(n, K, device="cuda", dtype=torch.float16).to(dt)
        xf = torch.randn(n, K, device="cuda", dtype=torch.float16)
        xb = xf.bfloat16()
        attempt(f"_scaled_mm fp8 x fp8 n={n}", lambda: torch._scaled_mm(x8, w.t(), scale_a=sc, scale_b=sc, out_dtype=torch.float16))
        attempt(f"_scaled_mm f16 act n={n}", lambda: torch._scaled_mm(xf, w.t(), scale_a=sc, scale_b=sc, out_dtype=torch.float16))
        attempt(f"_scaled_mm bf16 act n={n}", lambda: torch._scaled_mm(xb, w.t(), scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16))
        attempt(f"matmul f16 x fp8 n={n}", lambda: torch.matmul(xf, w.t()))
        attempt(f"matmul fp8 x fp8 n={n}", lambda: torch.matmul(x8, w.t()))
        attempt(f"F.linear f16 x fp8 n={n}", lambda: F.linear(xf, w))
        attempt(f"F.linear fp8 x fp8 n={n}", lambda: F.linear(x8, w))
    attempt("w.to(float16)", lambda: w.to(torch.float16))
    attempt("w.to(bfloat16)", lambda: w.to(torch.bfloat16))
    attempt("w.to(float16)*scale", lambda: w.to(torch.float16) * sc.half())
