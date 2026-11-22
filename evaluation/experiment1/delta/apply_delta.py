import argparse
import json
import sys

import torch


# ==========================
# DeltaUpdate: apply a parameter-wise delta to a base checkpoint
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


def _load_delta(delta_path: str, base_state: dict) -> dict:
    """
    Load a delta from either:

      - .pt: torch-saved tensor dictionary
      - .json: flattened parameter arrays
    """

    if delta_path.endswith(".pt"):
        obj = torch.load(delta_path, map_location="cpu")

        if not isinstance(obj, dict):
            raise TypeError(
                f"Unable to interpret {delta_path!r} as a delta state dict."
            )

        return {
            name: tensor.float()
            for name, tensor in obj.items()
        }

    if delta_path.endswith(".json"):
        with open(delta_path, "r") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise TypeError(
                f"Unable to interpret {delta_path!r} as a JSON delta."
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
            ).reshape(base_tensor.shape)

        return delta

    raise ValueError(
        f"Unsupported delta format: {delta_path!r}. "
        "Use .pt or .json."
    )


def apply_delta(base_path: str, delta_path: str):
    """
    Reconstruct:

        new = base + delta

    where:

        delta = new - old
    """

    checkpoint = _load_checkpoint(base_path)
    base_state = _extract_state_dict(checkpoint)

    delta = _load_delta(delta_path, base_state)

    base_keys = set(base_state.keys())
    delta_keys = set(delta.keys())

    only_base = sorted(base_keys - delta_keys)
    only_delta = sorted(delta_keys - base_keys)

    if only_base or only_delta:
        raise ValueError(
            "Parameter keys differ between base checkpoint and delta. "
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
            base_tensor.float() + delta_tensor.float()
        )

    # --------------------------------------------------
    # IMPORTANT:
    # Preserve the checkpoint structure.
    # --------------------------------------------------

    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = updated_state

    else:
        # Raw state_dict input: construct the structure
        # expected by the downstream loader.
        checkpoint = {
            "architecture": checkpoint.get("architecture"),
            "state_dict": updated_state,
        }

    # The required architecture key must exist.
    if "architecture" not in checkpoint:
        raise KeyError(
            "Base checkpoint is missing required key: 'architecture'. "
            "The apply script cannot infer the architecture automatically."
        )

    return checkpoint


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Apply a DeltaUpdate parameter-wise delta "
            "to a base model checkpoint."
        )
    )

    parser.add_argument(
        "base",
        help="Base/old model checkpoint (.pt)",
    )

    parser.add_argument(
        "delta",
        help="Delta file (.pt or .json)",
    )

    parser.add_argument(
        "--output",
        default="updated_model.pt",
        help="Output checkpoint",
    )

    args = parser.parse_args()

    print(f"Loading base checkpoint from: {args.base}")
    print(f"Loading delta from: {args.delta}")

    try:
        updated_checkpoint = apply_delta(
            args.base,
            args.delta,
        )

    except Exception as exc:
        print(f"Failed to apply delta: {exc}")
        sys.exit(2)

    state_dict = updated_checkpoint["state_dict"]

    n_values = sum(
        tensor.numel()
        for tensor in state_dict.values()
    )

    print(
        f"Applied delta to {len(state_dict)} parameters "
        f"({n_values} values total)."
    )

    torch.save(
        updated_checkpoint,
        args.output,
    )

    print(f"Saved updated checkpoint to: {args.output}")