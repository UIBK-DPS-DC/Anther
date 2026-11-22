import argparse
import struct
import sys

import torch


# ==========================
# DeltaUpdate baseline:
# raw parameter-wise diff between two model .pt checkpoints
# ==========================


def _load_state_dict(path: str) -> dict:
    """
    Load a raw parameter-name -> tensor mapping from a .pt file.

    Unlike svd_delta.py, this does NOT reconstruct a typed model — the
    DeltaUpdate baseline diffs whatever tensors are in the checkpoint
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


def serialize_delta(delta: dict[str, torch.Tensor]) -> bytes:
    """
    Serialize a parameter-wise delta into a custom binary format.

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
            float32 value 0
            float32 value 1
            ...

    The tensor values are always serialized as float32.
    """

    buf = bytearray()

    # Number of parameters
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
        # Tensor values
        # --------------------------------
        values = tensor.flatten().tolist()

        if len(values) > 0xFFFFFFFF:
            raise ValueError(
                f"Tensor {name!r} has too many values: {len(values)}"
            )

        buf.extend(struct.pack("<I", len(values)))

        if values:
            buf.extend(
                struct.pack(
                    f"<{len(values)}f",
                    *values,
                )
            )

    return bytes(buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "DeltaUpdate baseline: raw parameter-wise diff "
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
        default="delta_update.bin",
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

    # Serialize directly to the custom .bin format.
    serialized_data = serialize_delta(delta)

    with open(args.output, "wb") as f:
        f.write(serialized_data)

    print(
        f"Saved binary delta to: {args.output} "
        f"({len(serialized_data)} bytes)"
    )