# Model test 2: Llama 3.1 8B Instruct FP8 on the RX 7900 XT

Date: 2026-11-15. Model: RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 (Llama 3.1 Community License), compressed-tensors
`naive-quantized`, FP8 weights with one BF16 scale per tensor (checked in the safetensors header), `lm_head` BF16.
Converted with `bench/llama-cpp-convert-naive-fp8.patch` (unmodified llama.cpp asserts on `naive-quantized`).
Files: `models/llama31-8b-auto.gguf` (9.1 GB: 6.98 GB F8_E4M3, BF16 output and token embedding),
`llama31-8b-fp8-outq8.gguf` (8.6 GB, Q8_0 output layer via `proto/gguf_output_q8.py`), `llama31-8b-q8_0.gguf`
(8.5 GB, `convert_hf_to_gguf.py --outtype q8_0` from the same checkpoint). No F16 file (16 GB). Same method as
notes/model-test-1.md: `MODEL=llama31-8b bench/run-model.sh ppl kld bench batched`, run
`build/model-runs/20261115-175739-llama31-8b/`; device 0 only, shared GPU lock.

## In short

- Patched FP8 is faster than Q8_0 for one stream (+7 %), for prompts (+17 %) and for 2-7 parallel streams (+1 to +7 %);
  8 parallel streams are 8 % slower (the same open gap as on 3B).
- Unmodified llama.cpp: 2.7x slower than patched FP8 for one stream and NaN on prompts ("GGGG..." replies).
- Patched FP8 decode matches unmodified llama.cpp's exact path to rounding level (KLD 0.000002-0.000005); prompts
  0.000172; Q8_0 is 0.000544 from the same reference. Perplexity FP8 7.266-7.268, Q8_0 7.276.
- Real prompts: FP8 and Q8_0 give different wording (both proper answers); on 3B they were identical. Expected: Q8_0
  is a re-rounding of the FP8 weights and greedy decoding diverges at a near tie.

## Quality

Perplexity, 50 chunks of the fixed local text (MEASURED): patched FP8 as shipped 7.2683, FP8 with Q8_0 output 7.2655,
Q8_0 7.2763 (patched and unmodified identical); unmodified FP8 NaN in every chunk.

KL divergence, 10 chunks, against unmodified llama.cpp's FP8 with 8-token batches (its F32 decode path) (MEASURED):

| Patched run | Mean KLD | Max KLD | Same top token |
|---|---|---|---|
| 8-token batches (rows kernel) | 0.000002 | 0.000066 | 99.96 % |
| 3-token batches | 0.000004 | 0.000312 | 99.96 % |
| 1-token batches | 0.000005 | 0.000581 | 99.92 % |
| 512-token batches (WMMA prompt kernel) | 0.000172 | 0.001941 | 98.20 % |
| Q8_0 model, for scale | 0.000544 | 0.018476 | 98.51 % |

## Speed (MEASURED, medians of 3 rounds x 3 for llama-bench, 2 rounds for batched)

`llama-bench`, tokens/s:

| Model / build | pp512 | tg128 |
|---|---|---|
| Q8_0, patched | 3469.6 | 76.3 |
| FP8 with Q8_0 output, patched | 4048.5 | 81.9 |
| FP8 as shipped, patched | 4085.6 | 78.0 |
| FP8 with Q8_0 output, unmodified | 2976.4 (NaN output) | 30.3 |
| FP8 as shipped, unmodified | 3083.4 (NaN output) | 29.7 |

`llama-batched-bench`, decode tokens/s (all sequences):

| Sequences | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 76.3 | 140.9 | 203.8 | 266.8 | 323.6 | 374.0 | 415.0 | 451.8 |
| FP8 patched | 81.7 | 149.2 | 218.5 | 279.9 | 340.8 | 395.8 | 417.5 | 417.5 |
| FP8 / Q8_0 | 1.07 | 1.06 | 1.07 | 1.05 | 1.05 | 1.06 | 1.01 | 0.92 |
| FP8 unmodified | 30.4 | 52.5 | 74.9 | 95.2 | 113.8 | 128.9 | 125.8 | 138.0 |

FP8 gives the same total throughput at 8 sequences as at 7, so the 8-column setting (2 rows per warp, ksplit 2) is
the weak point here too.

## Real prompts (`build/prompt-test-8b/`)

Same prompts and settings as notes/prompt-test-1.md (`llama-completion -cnv -st`, `--temp 0 --seed 1`). Prompt 2 used
`-c 4096`; prompt 1 ran without `-c`, so llama.cpp sized the context to free VRAM (89,344 tokens). Greedy output does
not depend on the context size.

Prompt 1 ("how a refrigerator keeps food cold"): FP8 and Q8_0 differ after the first sentence; both are simplified
explanations of the refrigerant cycle with some physics detail wrong (FP8: the gas "releases its heat to the air"
inside the fridge; Q8_0: the refrigerant turns to gas "when it gets cold"). Unmodified FP8: "GGGG...".

Prompt 2 (Python prime check): identical, correct code; the explanation differs in two words ("larger than that" vs
"larger than the square root"). Unmodified FP8: "GGGG...".

FP8 reply, prompt 2:

````
Here is a short Python function that checks whether a number is prime:
```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```
This function works by checking divisibility of the input number `n` up to its square root, as any factor larger than
that would have a corresponding factor smaller than the square root.
````

## Incident

A Sonnet worker writing this up was killed by a system-wide out-of-memory event at 18:13. INFERRED cause: its
`llama-completion` runs had no `-c`, so llama.cpp sized the context to free VRAM (89,344 tokens; about 11 GB of KV
cache for 8B), on a machine whose swap was already full. The lead finished the work with `-c 4096`.
