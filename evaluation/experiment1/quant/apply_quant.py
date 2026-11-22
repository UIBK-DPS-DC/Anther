import argparse
import json
import struct
import sys

import torch


# ==========================
# QuantDelta / DeltaUpdate:
# apply a parameter-wise delta to a base checkpoint
# ==========================


def _load_checkpoint(path: str):
    """
    Load a checkpoint from a .pt file.

    Expected checkpoint structure:

        {
            "architecture": ...,
            "state_dict": {...}
        }

    Also accepts a raw state_dict for compatibility.
    """

    if not path.endswith(".pt"):
        raise ValueError(
            f"Unsupported checkpoint format: {path!r}. "
            "Only .pt files are supported."
        )

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "state_dict"):
        return {
            "state_dict": obj.state_dict()
        }

    raise TypeError(
        f"Unable to interpret {path!r} as a checkpoint."
    )


def _extract_state_dict(checkpoint: dict) -> dict:
    """
    Extract the actual parameter state dict.
    """

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

        if not isinstance(state_dict, dict):
            raise TypeError(
                "Checkpoint['state_dict'] is not a dictionary."
            )

        return state_dict

    # Compatibility with raw state_dict checkpoints
    return checkpoint


def _read_exact(f, size: int) -> bytes:
    """
    Read exactly `size` bytes from a binary file.
    """

    data = f.read(size)

    if len(data) != size:
        raise ValueError(
            f"Unexpected end of binary delta file. "
            f"Expected {size} bytes, got {len(data)}."
        )

    return data


def _read_uint16(f) -> int:
    return struct.unpack(
        "<H",
        _read_exact(f, 2),
    )[0]


def _read_uint32(f) -> int:
    return struct.unpack(
        "<I",
        _read_exact(f, 4),
    )[0]


def _read_uint8(f) -> int:
    return struct.unpack(
        "<B",
        _read_exact(f, 1),
    )[0]


def _read_float32(f) -> float:
    return struct.unpack(
        "<f",
        _read_exact(f, 4),
    )[0]


def _read_int32(f) -> int:
    return struct.unpack(
        "<i",
        _read_exact(f, 4),
    )[0]


def _load_quant_delta_bin(
    delta_path: str,
    base_state: dict,
) -> dict:
    """
    Load a QuantDelta .bin file.

    Binary format:

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

    Values are dequantized using:

        delta = (q - zero_point) * scale
    """

    delta = {}

    with open(delta_path, "rb") as f:

        # --------------------------------
        # Number of parameters
        # --------------------------------
        num_parameters = _read_uint16(f)

        for parameter_index in range(num_parameters):

            # --------------------------------
            # Parameter name
            # --------------------------------
            name_length = _read_uint16(f)

            name_bytes = _read_exact(
                f,
                name_length,
            )

            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Invalid UTF-8 parameter name at "
                    f"parameter index {parameter_index}."
                ) from exc

            # --------------------------------
            # Check parameter exists
            # --------------------------------
            if name not in base_state:
                raise ValueError(
                    f"QuantDelta contains parameter '{name}', "
                    "but it does not exist in the base checkpoint."
                )

            base_tensor = base_state[name]

            if not isinstance(base_tensor, torch.Tensor):
                raise TypeError(
                    f"Base parameter '{name}' is not a tensor."
                )

            # --------------------------------
            # Tensor shape
            # --------------------------------
            ndim = _read_uint8(f)

            shape = tuple(
                _read_uint32(f)
                for _ in range(ndim)
            )

            # --------------------------------
            # Number of values
            # --------------------------------
            num_values = _read_uint32(f)

            # --------------------------------
            # Quantization parameters
            # --------------------------------
            scale = _read_float32(f)
            zero_point = _read_int32(f)

            # --------------------------------
            # Validate shape
            # --------------------------------
            expected_num_values = 1

            for dim in shape:
                expected_num_values *= dim

            if expected_num_values != num_values:
                raise ValueError(
                    f"Invalid tensor metadata for '{name}': "
                    f"shape={shape} implies "
                    f"{expected_num_values} values, "
                    f"but file contains {num_values}."
                )

            if tuple(base_tensor.shape) != shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"base={tuple(base_tensor.shape)}, "
                    f"delta={shape}"
                )

            # --------------------------------
            # Read quantized values
            # --------------------------------
            raw_values = _read_exact(
                f,
                num_values,
            )

            q = torch.tensor(
                list(raw_values),
                dtype=torch.float32,
            )

            # --------------------------------
            # Dequantize
            # --------------------------------
            reconstructed_delta = (
                q - zero_point
            ) * scale

            reconstructed_delta = reconstructed_delta.reshape(
                shape
            )

            delta[name] = reconstructed_delta

    return delta


def _load_delta(
    delta_path: str,
    base_state: dict,
) -> dict:
    """
    Load a delta from:

      - .pt  : raw floating-point tensor dictionary
      - .json: flattened parameter arrays
      - .bin : QuantDelta 8-bit quantized binary format
    """

    # ==============================================
    # Original floating-point .pt delta
    # ==============================================

    if delta_path.endswith(".pt"):

        obj = torch.load(
            delta_path,
            map_location="cpu",
        )

        if not isinstance(obj, dict):
            raise TypeError(
                f"Unable to interpret {delta_path!r} "
                "as a delta state dict."
            )

        return {
            name: tensor.float()
            for name, tensor in obj.items()
        }

    # ==============================================
    # Original JSON delta
    # ==============================================

    if delta_path.endswith(".json"):

        with open(delta_path, "r") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise TypeError(
                f"Unable to interpret {delta_path!r} "
                "as a JSON delta."
            )

        delta = {}

        for name, values in obj.items():

            if name not in base_state:
                raise ValueError(
                    f"Delta contains parameter '{name}', "
                    "but it does not exist in the base checkpoint."
                )

            base_tensor = base_state[name]

            delta[name] = torch.tensor(
                values,
                dtype=torch.float32,
            ).reshape(
                base_tensor.shape
            )

        return delta

    # ==============================================
    # QuantDelta binary
    # ==============================================

    if delta_path.endswith(".bin"):

        return _load_quant_delta_bin(
            delta_path,
            base_state,
        )

    raise ValueError(
        f"Unsupported delta format: {delta_path!r}. "
        "Use .pt, .json, or QuantDelta .bin."
    )


def apply_delta(
    base_path: str,
    delta_path: str,
):
    """
    Reconstruct:

        new = base + delta

    For QuantDelta:

        delta ≈ dequantized_quantized_delta

    Therefore:

        new ≈ base + dequantized_delta
    """

    checkpoint = _load_checkpoint(
        base_path
    )

    base_state = _extract_state_dict(
        checkpoint
    )

    delta = _load_delta(
        delta_path,
        base_state,
    )

    base_keys = set(
        base_state.keys()
    )

    delta_keys = set(
        delta.keys()
    )

    only_base = sorted(
        base_keys - delta_keys
    )

    only_delta = sorted(
        delta_keys - base_keys
    )

    if only_base or only_delta:
        raise ValueError(
            "Parameter keys differ between base checkpoint "
            "and delta. "
            f"only_base={only_base}, "
            f"only_delta={only_delta}"
        )

    updated_state = {}

    for name in base_state:

        base_tensor = base_state[name]
        delta_tensor = delta[name]

        if not isinstance(base_tensor, torch.Tensor):
            raise TypeError(
                f"Base parameter '{name}' is not a tensor."
            )

        if base_tensor.shape != delta_tensor.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': "
                f"base={tuple(base_tensor.shape)}, "
                f"delta={tuple(delta_tensor.shape)}"
            )

        updated_state[name] = (
            base_tensor.float()
            + delta_tensor.float()
        )

    # --------------------------------------------------
    # Preserve checkpoint structure
    # --------------------------------------------------

    if "state_dict" in checkpoint:

        checkpoint["state_dict"] = updated_state

    else:

        checkpoint = {
            "architecture": checkpoint.get(
                "architecture"
            ),
            "state_dict": updated_state,
        }

    # --------------------------------------------------
    # Architecture is required
    # --------------------------------------------------

    if "architecture" not in checkpoint:

        raise KeyError(
            "Base checkpoint is missing required key: "
            "'architecture'. The apply script cannot "
            "infer the architecture automatically."
        )

    return checkpoint


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Apply a DeltaUpdate or QuantDelta delta "
            "to a base model checkpoint."
        )
    )

    parser.add_argument(
        "base",
        help="Base/old model checkpoint (.pt)",
    )

    parser.add_argument(
        "delta",
        help="Delta file (.pt, .json, or QuantDelta .bin)",
    )

    parser.add_argument(
        "--output",
        default="updated_model.pt",
        help="Output checkpoint",
    )

    args = parser.parse_args()

    print(
        f"Loading base checkpoint from: "
        f"{args.base}"
    )

    print(
        f"Loading delta from: "
        f"{args.delta}"
    )

    try:

        updated_checkpoint = apply_delta(
            args.base,
            args.delta,
        )

    except Exception as exc:

        print(
            f"Failed to apply delta: {exc}"
        )

        sys.exit(2)

    state_dict = (
        updated_checkpoint["state_dict"]
    )

    n_values = sum(
        tensor.numel()
        for tensor in state_dict.values()
    )

    print(
        f"Applied delta to "
        f"{len(state_dict)} parameters "
        f"({n_values} values total)."
    )

    torch.save(
        updated_checkpoint,
        args.output,
    )

    print(
        f"Saved updated checkpoint to: "
        f"{args.output}"
    )