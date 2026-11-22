"""
extract_lora_update.py

Turn a trained LoRA checkpoint (produced with the LoRA-enabled models.py)
into a tiny quantized "update" binary for a TinyML end-device that already
holds the frozen base model.

Either path is followed by INT8 (or INT4) quantization of the resulting
A/B factors, packed into a compact custom binary (`.lrq`) that a C/C++
firmware can memory-map and dequantize with ~10 lines of code.

The `compare` command quantizes the *entire* dense model the same way and
prints it side-by-side with the LoRA-delta binary, so you can show the
"LoRA + quant" vs "quant alone" size story directly.

Usage
-----
    # 1) Compare against quantizing the whole model:
    python lora_tinyml_export.py compare \
        --checkpoint runs/dscnn_finetuned.pt \
        --model-name DSCNN --lora-rank 4 --lora-alpha 4 \
        --lora-binary dscnn_delta_int8.lrq --bits 8

    # 2) (Alternative) Extract the difference between two *plain*
    #    dense checkpoints of the same architecture:
    python lora_tinyml_export.py extract \
        --base runs/pretrained_dense.pt --finetuned runs/finetuned_dense.pt \
        --rank 4 --bits 8 --out svd_delta_int8.lrq

    # 3) Sanity-check a .lrq file (dequantize + report reconstruction error
    #    against the original checkpoint):
    python lora_tinyml_export.py verify \
        --checkpoint runs/dscnn_finetuned.pt --lora-binary dscnn_delta_int8.lrq
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from typing import Dict, Tuple

import numpy as np
import torch

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import networkserver.models_lora as models


# ---------------------------------------------------------------------------
# Which parameter names count as "the LoRA delta" for this codebase.
#
# NOTE: this list is deliberately explicit rather than "contains 'lora'"
# because SVDF's adapters are named freq_A/freq_B/time_A/time_B (no "lora"
# substring) -- see the README note below about a bug this causes during
# training.
# ---------------------------------------------------------------------------
LORA_LEAF_NAMES = {
    "lora_A", "lora_B",                       # loralib Linear/Conv2d, LoRAConv1d
    "lora_A_ih", "lora_B_ih",                 # LoRAGRUCell (input-hidden)
    "lora_A_hh", "lora_B_hh",                 # LoRAGRUCell (hidden-hidden)
    "freq_A", "freq_B", "time_A", "time_B",   # SVDF adapters
}


def is_lora_param(name: str) -> bool:
    return name.split(".")[-1] in LORA_LEAF_NAMES


# ---------------------------------------------------------------------------
# Quantization (per-tensor, symmetric, INT8 or INT4)
# ---------------------------------------------------------------------------

def quantize_tensor(t: torch.Tensor, bits: int = 8) -> Tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization. Returns (int8 array, scale)."""
    arr = t.detach().cpu().numpy().astype(np.float32)
    qmax = 2 ** (bits - 1) - 1
    amax = float(np.abs(arr).max())
    scale = amax / qmax if amax > 0 else 1.0
    q = np.clip(np.round(arr / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def dequantize_tensor(q: np.ndarray, scale: float, shape) -> np.ndarray:
    return (q.astype(np.float32) * scale).reshape(shape)


def pack_int4(q_int8: np.ndarray) -> np.ndarray:
    """Pack a flat int8 array (values in [-8,7]) two-per-byte."""
    flat = q_int8.flatten().astype(np.int8)
    pad = flat.size % 2 == 1
    if pad:
        flat = np.append(flat, 0)
    lo = (flat[0::2].astype(np.uint8) & 0x0F)
    hi = (flat[1::2].astype(np.uint8) & 0x0F) << 4
    return (lo | hi).astype(np.uint8)


def unpack_int4(packed: np.ndarray, num_elements: int) -> np.ndarray:
    lo = (packed & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi)
    out = np.empty(packed.size * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:num_elements]


# ---------------------------------------------------------------------------
# Compact binary format ("LRQ1")
#
# [4]   magic = b"LRQ1"
# [1]   bits (8 or 4)
# [4]   meta_len (uint32, little-endian)
# [meta_len] meta json (utf-8) -- mode, num_tensors
# for each tensor:
#   [2]  name_len (uint16)
#   [name_len] name (utf-8)
#   [1]  ndim (uint8)
#   [ndim * 4] shape dims (uint32 each)
#   [4]  scale (float32)
#   [4]  data_len (uint32, bytes)
#   [data_len] quantized data (int8, or packed uint8 nibbles for 4-bit)
# ---------------------------------------------------------------------------
MAGIC = b"LRQ1"


def write_lora_binary(path: str, tensors: Dict[str, torch.Tensor], meta: dict, bits: int = 8) -> int:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", bits))
        meta_bytes = json.dumps(meta).encode("utf-8")
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            q, scale = quantize_tensor(t, bits)
            shape = list(t.shape)
            name_b = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<B", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<f", scale))
            data = q.tobytes() if bits == 8 else pack_int4(q).tobytes()
            f.write(struct.pack("<I", len(data)))
            f.write(data)
    return os.path.getsize(path)


def read_lora_binary(path: str):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Not a .lrq file: {path}"
        (bits,) = struct.unpack("<B", f.read(1))
        (meta_len,) = struct.unpack("<I", f.read(4))
        meta = json.loads(f.read(meta_len).decode("utf-8"))
        (n_tensors,) = struct.unpack("<I", f.read(4))
        tensors = {}
        for _ in range(n_tensors):
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode("utf-8")
            (ndim,) = struct.unpack("<B", f.read(1))
            shape = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
            (scale,) = struct.unpack("<f", f.read(4))
            (data_len,) = struct.unpack("<I", f.read(4))
            data = f.read(data_len)
            n_el = int(np.prod(shape)) if shape else 1
            if bits == 8:
                q = np.frombuffer(data, dtype=np.int8).reshape(shape)
            else:
                packed = np.frombuffer(data, dtype=np.uint8)
                q = unpack_int4(packed, n_el).reshape(shape)
            tensors[name] = torch.from_numpy(dequantize_tensor(q, scale, shape).copy())
        return tensors, meta, bits


# ---------------------------------------------------------------------------
# "quantize the whole dense model" -- the baseline to compare against
# ---------------------------------------------------------------------------

def quantized_full_model_size(state_dict: Dict[str, torch.Tensor], bits: int = 8) -> int:
    """Bytes needed to store every float parameter at the given bit width,
    using the same per-tensor symmetric scheme (scale + packed data), no
    LoRA involved -- i.e. "quantization alone"."""
    total = 0
    for t in state_dict.values():
        if t.dtype not in (torch.float32, torch.float16, torch.float64):
            continue
        n = t.numel()
        data_bytes = n if bits == 8 else (n + 1) // 2
        total += 4 + data_bytes  # scale (fp32) + packed data
    return total


def fp32_model_size(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * 4 for t in state_dict.values() if t.is_floating_point())


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_extract(args):
    """
    Compress the LoRA delta between two consecutive version checkpoints.

    Only LoRA parameters are considered. Dense/base weights are ignored
    entirely because they are guaranteed to be identical between versions.
    LoRA deltas are quantized directly.
    """
    base_sd = torch.load(args.base, map_location="cpu")
    ft_sd = torch.load(args.finetuned, map_location="cpu")

    if "state_dict" in base_sd:
        base_sd = base_sd["state_dict"]
    if "state_dict" in ft_sd:
        ft_sd = ft_sd["state_dict"]

    tensors = {}
    changed = 0

    for name, w_ft in ft_sd.items():
        # Ignore everything except LoRA parameters.
        if not is_lora_param(name):
            continue

        if not w_ft.is_floating_point():
            continue

        if name not in base_sd:
            # LoRA parameter did not exist in the previous checkpoint.
            # Store its full value as the delta.
            tensors[f"{name}.delta"] = w_ft
            changed += 1
            continue

        w_base = base_sd[name]

        if w_base.shape != w_ft.shape:
            continue

        delta = w_ft - w_base

        # No change to this LoRA parameter.
        if float(delta.abs().max()) == 0.0:
            continue

        tensors[f"{name}.delta"] = delta
        changed += 1

    if not tensors:
        raise SystemExit("No changed LoRA tensors found between the two checkpoints.")

    meta = {
        "mode": "lora_delta",
        "num_tensors": len(tensors),
        "changed_lora_tensors": changed,
    }

    size = write_lora_binary(
        args.out,
        tensors,
        meta,
        bits=args.bits,
    )

    print(
        f"{changed} LoRA tensor(s) changed and were quantized directly"
    )
    print(f"Wrote {args.out}  ({size:,} bytes, INT{args.bits})")


def cmd_compare(args):
    model = models.build_model(args.model_name, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    full_sd = model.state_dict()

    fp32_bytes = fp32_model_size(full_sd)
    quant_full_bytes = quantized_full_model_size(full_sd, bits=args.bits)
    lora_bytes = os.path.getsize(args.lora_binary)

    print("Size comparison")
    print("-" * 50)
    print(f"  Full model, FP32                : {fp32_bytes:>10,} bytes")
    print(f"  Full model, INT{args.bits} quantized only  : {quant_full_bytes:>10,} bytes  "
          f"({fp32_bytes / quant_full_bytes:.1f}x smaller than FP32)")
    print(f"  LoRA delta, INT{args.bits} quantized (.lrq): {lora_bytes:>10,} bytes  "
          f"({fp32_bytes / lora_bytes:.1f}x smaller than FP32, "
          f"{quant_full_bytes / lora_bytes:.1f}x smaller than full-model quant)")


def cmd_verify(args):
    tensors, meta, bits = read_lora_binary(args.lora_binary)

    print(
        f"Loaded {len(tensors)} tensors from {args.lora_binary} "
        f"(mode={meta.get('mode')}, INT{bits})"
    )

    if not args.base or not args.finetuned:
        return

    base_sd = torch.load(args.base, map_location="cpu")
    ft_sd = torch.load(args.finetuned, map_location="cpu")

    if "state_dict" in base_sd:
        base_sd = base_sd["state_dict"]
    if "state_dict" in ft_sd:
        ft_sd = ft_sd["state_dict"]

    max_err = 0.0
    mean_err = 0.0
    n = 0

    for name, delta_q in tensors.items():
        if not name.endswith(".delta"):
            continue

        lora_name = name[:-len(".delta")]

        if lora_name not in base_sd:
            print(f"[warn] {lora_name} missing from base checkpoint")
            continue

        if lora_name not in ft_sd:
            print(f"[warn] {lora_name} missing from finetuned checkpoint")
            continue

        reconstructed = base_sd[lora_name].float() + delta_q.float()
        target = ft_sd[lora_name].float()

        err = (reconstructed - target).abs()

        max_err = max(max_err, err.max().item())
        mean_err += err.mean().item()
        n += 1

    if n:
        print(f"Reconstruction error over {n} LoRA tensors:")
        print(f"  mean |err| = {mean_err / n:.6f}")
        print(f"  max  |err| = {max_err:.6f}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("extract", help="Extract the delta between two dense checkpoints")
    ps.add_argument("--base", required=True)
    ps.add_argument("--finetuned", required=True)
    ps.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ps.add_argument("--out", required=True)
    ps.set_defaults(func=cmd_extract)

    pc = sub.add_parser("compare", help="Compare LoRA-delta size vs quantizing the full model")
    pc.add_argument("--checkpoint", required=True)
    pc.add_argument("--model-name", required=True, choices=["MLP", "MLP_BIG", "CNN", "CNN1D", "DSCNN", "GRU", "CRNN", "SVDF"])
    pc.add_argument("--lora-rank", type=int, default=models.DEFAULT_LORA_RANK)
    pc.add_argument("--lora-alpha", type=int, default=models.DEFAULT_LORA_ALPHA)
    pc.add_argument("--bits", type=int, default=8, choices=[4, 8])
    pc.add_argument("--lora-binary", required=True)
    pc.set_defaults(func=cmd_compare)

    pv = sub.add_parser(
        "verify",
        help="Verify a quantized LoRA delta against base and finetuned checkpoints"
    )
    pv.add_argument("--lora-binary", required=True)
    pv.add_argument("--base", required=True)
    pv.add_argument("--finetuned", required=True)
    pv.set_defaults(func=cmd_verify)

    return p


def main():
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()