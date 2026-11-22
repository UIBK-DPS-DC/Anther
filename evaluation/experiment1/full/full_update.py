import argparse
import struct
import sys

import torch


# ==========================
# FullModel baseline:
# serialize a complete .pt checkpoint into a custom .bin format
# ==========================


def _load_state_dict(path: str) -> dict:
    """
    Load a raw parameter-name -> tensor mapping from a .pt file.

    This does NOT reconstruct a typed model. It extracts the state_dict
    directly from the checkpoint and serializes the tensors keyed by
    their original state_dict names.
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


def serialize_model(state_dict: dict) -> bytes:
    """
    Serialize a complete model into a custom binary format.

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

    This is intentionally compatible with the structure of the
    DeltaUpdate binary format, except that the stored tensor values
    are the complete model parameters rather than parameter deltas.
    """

    buf = bytearray()

    # --------------------------------
    # Number of parameters
    # --------------------------------
    if len(state_dict) > 0xFFFF:
        raise ValueError(
            f"Too many parameters ({len(state_dict)}). "
            "Maximum supported by this format is 65535."
        )

    buf.extend(struct.pack("<H", len(state_dict)))

    # --------------------------------
    # Serialize each parameter
    # --------------------------------
    for name, tensor in state_dict.items():

        # --------------------------------
        # Parameter name
        # --------------------------------
        name_bytes = name.encode("utf-8")

        if len(name_bytes) > 0xFFFF:
            raise ValueError(
                f"Parameter name is too long "
                f"({len(name_bytes)} bytes): {name!r}"
            )

        buf.extend(struct.pack("<H", len(name_bytes)))
        buf.extend(name_bytes)

        # --------------------------------
        # Tensor shape
        # --------------------------------
        tensor = (
            torch.as_tensor(tensor)
            .detach()
            .cpu()
            .float()
            .contiguous()
        )

        shape = tuple(tensor.shape)

        if len(shape) > 0xFF:
            raise ValueError(
                f"Tensor {name!r} has too many dimensions: "
                f"{len(shape)}"
            )

        buf.extend(struct.pack("<B", len(shape)))

        for dim in shape:
            if dim > 0xFFFFFFFF:
                raise ValueError(
                    f"Dimension too large for tensor {name!r}: {dim}"
                )

            buf.extend(struct.pack("<I", dim))

        # --------------------------------
        # Tensor values
        # --------------------------------
        n_values = tensor.numel()

        if n_values > 0xFFFFFFFF:
            raise ValueError(
                f"Tensor {name!r} has too many values: {n_values}"
            )

        buf.extend(struct.pack("<I", n_values))

        if n_values > 0:
            values = tensor.flatten().tolist()

            buf.extend(
                struct.pack(
                    f"<{n_values}f",
                    *values,
                )
            )

    return bytes(buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Full model serializer: convert a .pt checkpoint "
            "into a custom .bin model file"
        )
    )

    parser.add_argument(
        "model",
        help="Model .pt path",
    )

    parser.add_argument(
        "--output",
        default="model_version.bin",
        help="Output binary file",
    )

    args = parser.parse_args()

    print(f"Loading model checkpoint from: {args.model}")

    try:
        state_dict = _load_state_dict(args.model)
    except Exception as exc:
        print(f"Failed to load checkpoint: {exc}")
        sys.exit(2)

    n_values = sum(
        torch.as_tensor(tensor).numel()
        for tensor in state_dict.values()
    )

    print(
        f"Loaded model with {len(state_dict)} parameters "
        f"({n_values} values total)."
    )

    print("Serializing full model as float32...")

    try:
        serialized_data = serialize_model(state_dict)
    except Exception as exc:
        print(f"Failed to serialize model: {exc}")
        sys.exit(2)

    try:
        with open(args.output, "wb") as f:
            f.write(serialized_data)
    except Exception as exc:
        print(f"Failed to write output file: {exc}")
        sys.exit(2)

    print(
        f"Saved full model to: {args.output} "
        f"({len(serialized_data)} bytes)"
    )