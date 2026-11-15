"""Copy a GGUF file, re-quantizing only output.weight (BF16/F16/F32) to Q8_0; everything else is copied byte for byte.

Used to give an FP8 model the same Q8_0 output layer as the Q8_0 model it is compared with, so the speed comparison
measures the FP8 weights, not the 16-bit output layer the FP8 checkpoint ships with.

Usage: python -I proto/gguf_output_q8.py <in.gguf> <out.gguf>   (needs build/llama.cpp/gguf-py on sys.path)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build/llama.cpp/gguf-py"))
import gguf  # noqa: E402

T = gguf.GGMLQuantizationType


def to_f32(tensor) -> np.ndarray:
    # GGUFReader exposes BF16 as raw bytes (uint8, two per value)
    if tensor.tensor_type == T.BF16:
        bits = np.ascontiguousarray(tensor.data).view(np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32)
    if tensor.tensor_type in (T.F16, T.F32):
        return tensor.data.astype(np.float32)
    raise SystemExit(f"{tensor.name}: unsupported source type {tensor.tensor_type.name}")


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    data = {}
    for t in reader.tensors:
        if t.name == "output.weight":
            rows, cols = int(t.shape[1]), int(t.shape[0])
            q = gguf.quants.quantize(to_f32(t).reshape(rows, cols), T.Q8_0)
            data[t.name] = q
            writer.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, T.Q8_0)
            print(f"{t.name}: {t.tensor_type.name} {t.n_bytes / 1e6:.0f} MB -> Q8_0 {q.nbytes / 1e6:.0f} MB")
        else:
            data[t.name] = t.data
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    if "output.weight" not in data:
        raise SystemExit("no output.weight tensor (tied embeddings?)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(data[t.name], tensor_endianess=reader.endianess)
    writer.close()


if __name__ == "__main__":
    main()
