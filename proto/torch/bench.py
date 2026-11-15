import os, sys, json
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
import torch.nn.functional as F
arch = torch.cuda.get_device_properties(0).gcnArchName
if not arch.startswith("gfx1100"):
    sys.exit("not gfx1100: " + arch)
M, K = 16384, 14336
SCALE = 0.5  # power of two: scale multiply is exact in fp16/bf16
torch.manual_seed(0)
w8 = (torch.randn(M, K, device="cuda", dtype=torch.float16)).to(torch.float8_e4m3fn)
w16 = w8.to(torch.float16) * SCALE
wbf = w8.to(torch.bfloat16) * SCALE
scale16 = torch.tensor(SCALE, device="cuda", dtype=torch.float16)
ref_w = (w8[:64].cpu().to(torch.float64)) * SCALE  # exact fp8 value * scale, float64
torch.cuda.synchronize()

def timeit(fn, warm, iters):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000.0 for s, e in ev)
    return t[len(t)//2], t[0], t[-1]

def relerr(out, x):
    ref = x[:, :].cpu().to(torch.float64) @ ref_w.t()
    got = out[:, :64].cpu().to(torch.float64)
    return ((got - ref).abs().max() / ref.abs().max()).item()

rows = []
for n in [1,2,3,4,5,6,7,8,512]:
    xf = torch.randn(n, K, device="cuda", dtype=torch.float16)
    xb = xf.to(torch.bfloat16)
    warm, iters = (20, 20) if n == 512 else (20, 200)
    cases = {
      "a_f16":  (lambda: F.linear(xf, w16), xf, 470.0),
      "b_bf16": (lambda: F.linear(xb, wbf), xb, 470.0),
      "c_fp8_convert": (lambda: F.linear(xf, w8.to(torch.float16) * scale16), xf, 235.0),
    }
    for name, (fn, xin, _) in cases.items():
        err = relerr(fn(), xin)
        med, mn, mx = timeit(fn, warm, iters)
        wbytes = M*K*(1 if name.startswith("c") else 2)
        row = dict(n=n, case=name, med=med, min=mn, max=mx, err=err,
                   gbps_stored=wbytes/med/1e3)
        print(json.dumps(row), flush=True)
        rows.append(row)
