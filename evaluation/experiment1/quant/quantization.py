import argparse
import struct
import sys

import torch


# ==========================
# QuantDelta baseline:
# per-tensor affine (uint8) quantized diff between two model .pt checkpoints
#
# This is the quantization counterpart to svd_delta.py: instead of a low-rank
# SVD approximation of each weight delta, each delta tensor is linearly
# quantized to 8 bits using a single (scale, zero_point) pair, following the
# standard post-training quantization formula:
#
#     wq = round(wf / s) + zero_point
#
# with s chosen so that the full [min, max] range of the delta tensor maps
# into [0, 255].
# ==========================


def _load_state_dict(path: str) -> dict:
    """
    Load a raw parameter-name -> tensor mapping from a .pt file.

    Unlike svd_delta.py, this does NOT reconstruct a typed model — the
    QuantDelta baseline diffs whatever tensors are in the checkpoint
    directly, keyed by their original state_dict names, with no reshaping.
    """
    if not path.endswith(".pt"):
        raise ValueError(
            f"Unsupported model file format: {path!r}. Only .pt files are supported."
        )

    obj = torch.load(path, map_location="cpu")

    if hasattr(obj, "state_dict"):
        return obj.state_dict()

    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]

        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]

        return obj

    raise TypeError(
        f"Unable to interpret {path!r} as a state dict "
        f"or model with .state_dict()."
    )


def compute_delta(old_path: str, new_path: str) -> dict:
    old_state = _load_state_dict(old_path)
    new_state = _load_state_dict(new_path)

    if old_state.keys() != new_state.keys():
        only_old = sorted(old_state.keys() - new_state.keys())
        only_new = sorted(new_state.keys() - old_state.keys())

        raise ValueError(
            f"Parameter keys differ between checkpoints. "
            f"only_old={only_old}, only_new={only_new}"
        )

    delta = {}

    for name in old_state:
        old_t = old_state[name].float()
        new_t = new_state[name].float()

        if old_t.shape != new_t.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': "
                f"old={tuple(old_t.shape)}, new={tuple(new_t.shape)}"
            )

        delta[name] = new_t - old_t

    return delta


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 4):
    """
    Per-tensor affine (asymmetric) quantization to an unsigned integer range.

    Maps [min(tensor), max(tensor)] -> [0, qmax] using:

        scale      = (max - min) / qmax
        zero_point = round(-min / scale)          (clamped to [0, qmax])
        q          = clip(round(w / scale) + zero_point, 0, qmax)

    Dequantization is:

        w_hat = (q - zero_point) * scale

    Returns (q_values: uint8 tensor, scale: float, zero_point: int).
    """

    qmax = (1 << num_bits) - 1  # 255 for 8 bits

    w_min = float(tensor.min())
    w_max = float(tensor.max())

    # Degenerate case: all-zero (or perfectly flat) delta. Avoid div-by-zero;
    # every value quantizes to 0 and dequantizes back to exactly 0.0.
    if w_max == w_min:
        scale = 1.0
        zero_point = 0
        q = torch.zeros_like(tensor, dtype=torch.uint8)
        return q, scale, zero_point

    scale = (w_max - w_min) / qmax

    zero_point = round(-w_min / scale)
    zero_point = max(0, min(qmax, zero_point))

    q = torch.round(tensor / scale) + zero_point
    q = torch.clamp(q, 0, qmax).to(torch.uint8)

    return q, scale, zero_point


def dequantize_tensor(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q.float() - zero_point) * scale


def serialize_quant_delta(delta: dict[str, torch.Tensor], num_bits: int = 8) -> bytes:
    """
    Serialize a per-tensor quantized delta into a custom binary format.

    Binary format (little-endian):

        uint16  number_of_parameters

        For each parameter:
            uint16  name_length
            bytes   UTF-8 parameter name

            uint8   number_of_dimensions
            uint32  dimension 0
            uint32  dimension 1
            ...

            uint32  number_of_values
            float32 scale
            int32   zero_point
            uint8   value 0
            uint8   value 1
            ...

    Unlike the DeltaUpdate baseline (which stores each value as float32),
    every tensor value here is a single quantized byte; only the per-tensor
    (scale, zero_point) pair carries floating-point precision.
    """

    if num_bits != 8:
        raise ValueError(
            "Only 8-bit quantization is currently supported by this format."
        )

    buf = bytearray()

    if len(delta) > 0xFFFF:
        raise ValueError(
            f"Too many parameters ({len(delta)}). "
            "Maximum supported by this format is 65535."
        )

    buf.extend(struct.pack("<H", len(delta)))

    for name, tensor in delta.items():
        # --------------------------------
        # Parameter name
        # --------------------------------
        name_bytes = name.encode("utf-8")

        if len(name_bytes) > 0xFFFF:
            raise ValueError(
                f"Parameter name is too long ({len(name_bytes)} bytes): {name!r}"
            )

        buf.extend(struct.pack("<H", len(name_bytes)))
        buf.extend(name_bytes)

        # --------------------------------
        # Tensor shape
        # --------------------------------
        tensor = torch.as_tensor(tensor).detach().cpu().float()

        shape = tuple(tensor.shape)

        if len(shape) > 0xFF:
            raise ValueError(
                f"Tensor {name!r} has too many dimensions: {len(shape)}"
            )

        buf.extend(struct.pack("<B", len(shape)))

        for dim in shape:
            buf.extend(struct.pack("<I", dim))

        # --------------------------------
        # Quantize + write values
        # --------------------------------
        flat = tensor.flatten()

        if flat.numel() > 0xFFFFFFFF:
            raise ValueError(
                f"Tensor {name!r} has too many values: {flat.numel()}"
            )

        buf.extend(struct.pack("<I", flat.numel()))

        q, scale, zero_point = quantize_tensor(flat, num_bits=num_bits)

        buf.extend(struct.pack("<f", scale))
        buf.extend(struct.pack("<i", zero_point))

        buf.extend(q.numpy().tobytes())

    return bytes(buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "QuantDelta baseline: per-tensor 8-bit affine-quantized diff "
            "between two model .pt checkpoints"
        )
    )

    parser.add_argument(
        "old",
        help="Old model .pt path",
    )

    parser.add_argument(
        "new",
        help="New model .pt path",
    )

    parser.add_argument(
        "--output",
        default="quant_delta.bin",
        help="Output binary file",
    )

    args = parser.parse_args()

    print(f"Loading old checkpoint from: {args.old}")
    print(f"Loading new checkpoint from: {args.new}")

    try:
        delta = compute_delta(args.old, args.new)
    except Exception as exc:
        print(f"Failed to compute delta: {exc}")
        sys.exit(2)

    n_values = sum(t.numel() for t in delta.values())

    print(
        f"Computed delta for {len(delta)} parameters "
        f"({n_values} values total)."
    )

    # Report per-tensor quantization error before serializing, so it's easy
    # to sanity-check quantization quality against the SVD baseline's
    # "actual information retained" logging.
    for name, tensor in delta.items():
        flat = tensor.flatten()
        q, scale, zero_point = quantize_tensor(flat)
        recon = dequantize_tensor(q, scale, zero_point)

        max_abs_err = float((flat - recon).abs().max()) if flat.numel() else 0.0

        print(
            f"  {name}: n={flat.numel():6d}  "
            f"scale={scale:.6e}  zero_point={zero_point:3d}  "
            f"max_abs_err={max_abs_err:.6e}"
        )

    serialized_data = serialize_quant_delta(delta)

    with open(args.output, "wb") as f:
        f.write(serialized_data)

    print(
        f"Saved binary quantized delta to: {args.output} "
        f"({len(serialized_data)} bytes)"
    )