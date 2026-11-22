import os
import struct
import torch
import numpy as np

from models import FlattenableModule, MLP, CNN, CNN1D, DSCNN, SVDF, GRUModel, CRNN

# Maps a model-type directory name (e.g. "ds_cnn") to its class.
MODEL_REGISTRY = {
    "mlp": MLP,
    "cnn": CNN,
    "cnn1d": CNN1D,
    "ds_cnn": DSCNN,
    "svdf": SVDF,
    "gru": GRUModel,
    "crnn": CRNN,
}

# Constructor kwargs per model type. Defaults match each class's __init__
# defaults. Override here (or pass model_kwargs into decompose()) if any
# training run used non-default hyperparameters (n_mels, n_classes, etc.) —
# a mismatch here will surface as a load_state_dict shape error, not a
# silent misload.
DEFAULT_MODEL_KWARGS = {
    "mlp": {"n_mels": 40, "n_frames": 101, "n_classes": 32},
    "cnn": {"n_classes": 32},
    "cnn1d": {"n_mels": 40, "n_classes": 32},
    "ds_cnn": {"n_classes": 32},
    "svdf": {"n_mels": 40, "n_frames": 101, "rank": 2, "units": 32, "n_classes": 32},
    "gru": {"n_mels": 40, "n_classes": 32},
    "crnn": {"n_classes": 32},
}


# Update types
TYPE_SVD = 0
TYPE_DIRECT = 1


def serialize_svd_triplets(
    triplets: list[dict],
    compression_level: int = 9,
) -> bytes:
    """
    Serialize SVD + direct updates into a compact compressed binary format.

    Format before compression:

        Header:
            4 bytes  magic = b"SVD1"
            1 byte   version
            2 bytes  number of updates

        Per update:
            1 byte   type
            2 bytes  matrix-name length
            N bytes  matrix name

        SVD:
            2 bytes  rank
            2 bytes  rows
            2 bytes  cols
            4 bytes  sigma (FP32)
            rows*2   u (FP16)
            cols*2   v (FP16)

        Direct:
            2 bytes  number of elements
            N*2      delta (FP16)

    The entire payload is then compressed using zlib.
    """

    buf = bytearray()

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    buf.extend(b"SVD1")
    buf.extend(struct.pack("<B", 1))  # format version
    buf.extend(struct.pack("<H", len(triplets)))

    # ------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------

    for triplet in triplets:

        matrix = triplet["matrix"]
        matrix_bytes = matrix.encode("utf-8")

        if len(matrix_bytes) > 65535:
            raise ValueError(
                f"Matrix name too long: {matrix}"
            )

        # Matrix name
        buf.extend(
            struct.pack(
                "<H",
                len(matrix_bytes)
            )
        )

        buf.extend(matrix_bytes)

        update_type = triplet.get("type", "svd")

        # ========================================================
        # SVD UPDATE
        # ========================================================

        if update_type == "svd":

            buf.extend(
                struct.pack(
                    "<B",
                    TYPE_SVD
                )
            )

            rank = triplet["rank"]
            sigma = float(triplet["sigma"])

            u = (
                torch.as_tensor(
                    triplet["u"]
                )
                .detach()
                .cpu()
                .flatten()
                .to(torch.float16)
            )

            v = (
                torch.as_tensor(
                    triplet["v"]
                )
                .detach()
                .cpu()
                .flatten()
                .to(torch.float16)
            )

            if u.numel() > 65535:
                raise ValueError(
                    f"U vector too large for {matrix}"
                )

            if v.numel() > 65535:
                raise ValueError(
                    f"V vector too large for {matrix}"
                )

            # rank
            buf.extend(
                struct.pack(
                    "<H",
                    rank
                )
            )

            # U/V dimensions
            buf.extend(
                struct.pack(
                    "<HH",
                    u.numel(),
                    v.numel()
                )
            )

            # sigma remains FP32
            buf.extend(
                struct.pack(
                    "<f",
                    sigma
                )
            )

            # FP16 U
            buf.extend(
                u.numpy().tobytes()
            )

            # FP16 V
            buf.extend(
                v.numpy().tobytes()
            )

        # ========================================================
        # DIRECT UPDATE
        # ========================================================

        elif update_type == "direct":

            buf.extend(
                struct.pack(
                    "<B",
                    TYPE_DIRECT
                )
            )

            delta = (
                torch.as_tensor(
                    triplet["delta"]
                )
                .detach()
                .cpu()
                .flatten()
                .to(torch.float16)
            )

            if delta.numel() > 65535:
                raise ValueError(
                    f"Direct update too large for {matrix}"
                )

            buf.extend(
                struct.pack(
                    "<H",
                    delta.numel()
                )
            )

            buf.extend(
                delta.numpy().tobytes()
            )

        else:
            raise ValueError(
                f"Unknown update type: {update_type!r}"
            )

    # ------------------------------------------------------------
    # Compress complete binary payload
    # ------------------------------------------------------------

    return bytes(buf)

class SVDDeltaDecomposer:
    """
    Converts the difference between two model versions into an ordered list
    of rank-1 SVD updates.

    flatten() returns a list of 2D weight matrices (one per logical layer),
    so the delta and SVD are computed independently per matrix, not over a
    single concatenated tensor.

    Each update:
        sigma * u * v^T

    applies to a specific matrix (identified by "matrix" index, matching the
    position in flatten()'s returned list) and can be independently
    transmitted and applied.
    """

    def decompose(
        self,
        old_model_path: str,
        new_model_path: str,
        svd_information: float,
        model_kwargs: dict | None = None,
    ):
        old_checkpoint = torch.load(
            old_model_path,
            map_location="cpu"
        )

        new_checkpoint = torch.load(
            new_model_path,
            map_location="cpu"
        )

        old_state = old_checkpoint["state_dict"]
        new_state = new_checkpoint["state_dict"]

        all_triplets = []

        for key in old_state:
            old_tensor = old_state[key]
            new_tensor = new_state[key]

            if old_tensor.shape != new_tensor.shape:
                raise ValueError(
                    f"{key}: old={old_tensor.shape}, new={new_tensor.shape}"
                )

            # Skip scalar buffers such as BatchNorm num_batches_tracked
            original_shape = tuple(old_tensor.shape)
            if old_tensor.ndim == 0:
                continue

            if old_tensor.ndim == 1:
                delta = (
                    new_tensor.float()
                    - old_tensor.float()
                )

                all_triplets.append(
                    {
                        "matrix": key,
                        "type": "direct",
                        "delta": delta.clone(),
                    }
                )

                continue

            elif old_tensor.ndim > 2:

                delta = (
                    new_tensor.float()
                    - old_tensor.float()
                ).reshape(
                    old_tensor.shape[0],
                    -1
                ).numpy()

            else:
                delta = (
                    new_tensor.float()
                    - old_tensor.float()
                ).numpy()
            
            U, S, Vh = np.linalg.svd(
                delta,
                full_matrices=False
            )

            # ----------------------------------------------------
            # Calculate total information for THIS matrix only
            # ----------------------------------------------------

            total_information = float(np.sum(S ** 2))

            if total_information == 0:
                print(
                    f"  Matrix {key}: "
                    f"zero delta, skipping"
                )
                continue

            target_information = svd_information * total_information

            # ----------------------------------------------------
            # Select ranks for THIS matrix
            # ----------------------------------------------------

            cumulative_information = 0.0
            selected_triplets = []

            for rank, sigma in enumerate(S):

                cumulative_information += float(sigma ** 2)

                selected_triplets.append(
                    {
                        "matrix": key,
                        "type": "svd",
                        "rank": rank,
                        "sigma": float(sigma),
                        "shape": original_shape,
                        "u": torch.from_numpy(U[:, rank].copy()),
                        "v": torch.from_numpy(Vh[rank, :].copy()),
                    }
                )

                if cumulative_information >= target_information:
                    break

            actual_information = (
                cumulative_information / total_information
            )

            print(
                f"  total rank: {len(S)}"
            )

            print(
                f"  selected rank: "
                f"{len(selected_triplets)}"
            )

            print(
                f"  requested information: "
                f"{svd_information:.4f}"
            )

            print(
                f"  actual information: "
                f"{actual_information:.4f}"
            )

            print(
                f"  U shape: {U.shape}"
            )

            print(
                f"  Vh shape: {Vh.shape}"
            )
            all_triplets.extend(selected_triplets)
        return all_triplets


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="SVD diff between two model .pt files")
    parser.add_argument("old", help="Old model .pt path")
    parser.add_argument("new", help="New model .pt path")
    parser.add_argument("--svd_information", type=float, default=1, help="Fraction of information to retain in SVD")
    parser.add_argument("--output", default="svd_triplets.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"Loading old model from: {args.old}")
    print(f"Loading new model from: {args.new}")

    decomposer = SVDDeltaDecomposer()

    try:
        triplets = decomposer.decompose(args.old, args.new, args.svd_information)
    except Exception as exc:
        print(f"Failed to compute SVD delta: {exc}")
        sys.exit(2)

    print(f"\nTotal triplets generated using {args.svd_information} information: {len(triplets)}")

    if not triplets:
        print("No differences found.")
        sys.exit(0)

    for t in triplets:

        if t.get("type") == "direct":

            delta = t["delta"]

            print(
                f"matrix={t['matrix']}, "
                f"type=direct, "
                f"elements={delta.numel()}"
            )

        else:

            u = t["u"]
            v = t["v"]

            print(
                f"matrix={t['matrix']}, "
                f"type=svd, "
                f"rank={t['rank']:3d}, "
                f"sigma={t['sigma']:.6e}, "
                f"u={len(u)}, "
                f"v={len(v)}"
            )

    serialized_data = serialize_svd_triplets(triplets)

    with open(args.output, "wb") as f:
        f.write(serialized_data)

    print(f"Saved {len(triplets)} SVD triplets to: {args.output}")