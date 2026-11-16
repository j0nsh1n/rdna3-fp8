# Prompt test 1: real replies, FP8 vs Q8_0

Date: 2026-11-15. Llama 3.2 3B Instruct (`models/llama32-3b-auto.gguf` = the FP8 file as converted, BF16 output layer;
`llama32-3b-q8_0.gguf`). `llama-completion -cnv -st -sys "You are a helpful assistant." -p <prompt> -n 300
--temp 0 --seed 1 -ngl 99`, device 0 only, under the GPU lock. Raw outputs and logs: `build/prompt-test/`.

## In short

- With the patch, FP8 and Q8_0 give word-for-word identical replies to both prompts, and the replies are correct.
- Unmodified llama.cpp with the same FP8 file prints only "GGGG...": the prompt goes through the large-batch path
  that produces NaN on RDNA 3 (notes/model-test-1.md), so the model never produces real text.
- Speeds printed by these single runs include model load and warm-up and are not used; see llama-bench results in
  notes/model-test-1.md.

## Prompt 1: "Explain in simple terms how a refrigerator keeps food cold. Use about five sentences."

Patched llama.cpp, FP8 (identical to Q8_0 byte for byte):

```
Here's a simple explanation:

A refrigerator keeps food cold by using a special liquid called refrigerant that changes state from liquid to gas as it absorbs heat from the inside of the fridge. This process is called evaporation. The refrigerant is pumped through a coil inside the fridge, where it absorbs heat from the air and the food. The refrigerant then carries the heat outside the fridge, where it releases it into the air. As the refrigerant cools down, it condenses back into a liquid and the cycle starts all over again, keeping the food cold. [end of text]
```

Unmodified llama.cpp, same FP8 file (first 60 characters of 300):

```
GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG...
```

## Prompt 2: "Write a short Python function that checks whether a number is prime, then explain in one sentence how it works."

Patched llama.cpp, FP8 (identical to Q8_0 byte for byte):

````
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

This function works by checking divisibility of the input number `n` by all numbers from 2 to the square root of `n`, and returning `False` as soon as it finds a divisor, or `True` if no divisors are found. [end of text]
````

Unmodified llama.cpp, same FP8 file:

```
GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG...
```
