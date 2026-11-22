#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import torch
import numpy as np


TYPE_SVD = 0
TYPE_DIRECT = 1

# Must match the writer in the serialization script exactly.
MAGIC = b"SVDQ"
VERSION = 2


def load_binary_updates(path):
    """
    Parse the binary format produced by serialize_svd_triplets():

        Header:
            4 bytes  magic = b"SVDQ"
            1 byte   version
            2 bytes  number of updates

        Per update:
            2 bytes  matrix-name length
            N bytes  matrix name
            1 byte   type (0 = SVD, 1 = direct)

        SVD:
            2 bytes  rank
            2 bytes  U length
            2 bytes  V length
            4 bytes  sigma (FP32)

            4 bytes  U scale (FP32)
            4 bytes  U zero_point (INT32)
            U_len    U (UINT8)

            4 bytes  V scale (FP32)
            4 bytes  V zero_point (INT32)
            V_len    V (UINT8)

        Direct:
            2 bytes  number of elements
            4 bytes  delta scale (FP32)
            4 bytes  delta zero_point (INT32)
            N bytes  delta (UINT8)

    Reconstruction (asymmetric affine dequantization):

        U ~= (Uq - U_zero_point) * U_scale
        V ~= (Vq - V_zero_point) * V_scale
        delta ~= (delta_q - delta_zero_point) * delta_scale
    """

    with open(path, "rb") as f:
        data = f.read()

    offset = 0

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    magic = data[offset:offset + 4]
    offset += 4

    if magic != MAGIC:
        raise ValueError(
            f"Bad magic bytes: {magic!r}, expected {MAGIC!r}"
        )

    version = struct.unpack_from("<B", data, offset)[0]
    offset += 1

    if version != VERSION:
        raise ValueError(
            f"Unsupported format version: {version}"
        )

    num_triplets = struct.unpack_from(
        "<H",
        data,
        offset,
    )[0]

    offset += 2

    updates = []

    for _ in range(num_triplets):

        # Matrix name length
        name_length = struct.unpack_from(
            "<H",
            data,
            offset,
        )[0]

        offset += 2

        # Matrix name
        matrix = data[
            offset:offset + name_length
        ].decode("utf-8")

        offset += name_length

        # Update type
        update_type = struct.unpack_from(
            "<B",
            data,
            offset,
        )[0]

        offset += 1

        # ========================================================
        # SVD UPDATE
        # ========================================================

        if update_type == TYPE_SVD:

            rank = struct.unpack_from(
                "<H",
                data,
                offset,
            )[0]

            offset += 2

            u_length, v_length = struct.unpack_from(
                "<HH",
                data,
                offset,
            )

            offset += 4

            sigma = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]

            offset += 4

            # ----------------------------------------------------
            # U: UINT8 + FP32 scale + INT32 zero_point
            #    -> dequantize to FP32
            # ----------------------------------------------------

            u_scale = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]

            offset += 4

            u_zero_point = struct.unpack_from(
                "<i",
                data,
                offset,
            )[0]

            offset += 4

            u_q = np.frombuffer(
                data,
                dtype=np.uint8,
                count=u_length,
                offset=offset,
            ).copy()

            offset += u_length

            u = (
                u_q.astype(np.float32) - u_zero_point
            ) * u_scale

            # ----------------------------------------------------
            # V: UINT8 + FP32 scale + INT32 zero_point
            #    -> dequantize to FP32
            # ----------------------------------------------------

            v_scale = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]

            offset += 4

            v_zero_point = struct.unpack_from(
                "<i",
                data,
                offset,
            )[0]

            offset += 4

            v_q = np.frombuffer(
                data,
                dtype=np.uint8,
                count=v_length,
                offset=offset,
            ).copy()

            offset += v_length

            v = (
                v_q.astype(np.float32) - v_zero_point
            ) * v_scale

            updates.append({
                "matrix": matrix,
                "type": "svd",
                "rank": rank,
                "sigma": sigma,
                "u": u,
                "v": v,
            })

        # ========================================================
        # DIRECT UPDATE
        # ========================================================

        elif update_type == TYPE_DIRECT:

            n_elements = struct.unpack_from(
                "<H",
                data,
                offset,
            )[0]

            offset += 2

            delta_scale = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]

            offset += 4

            delta_zero_point = struct.unpack_from(
                "<i",
                data,
                offset,
            )[0]

            offset += 4

            delta_q = np.frombuffer(
                data,
                dtype=np.uint8,
                count=n_elements,
                offset=offset,
            ).copy()

            offset += n_elements

            delta = (
                delta_q.astype(np.float32) - delta_zero_point
            ) * delta_scale

            updates.append({
                "matrix": matrix,
                "type": "direct",
                "delta": delta,
            })

        else:
            raise ValueError(
                f"Unknown update type byte: {update_type} "
                f"for matrix {matrix!r}"
            )

    return updates


def apply_updates(
    checkpoint,
    updates: list[dict],
):
    """
    Apply SVD and direct updates to a checkpoint.

    SVD updates are:

        ΔW = sigma * u * vᵀ

    Direct updates are:

        Δw = delta

    All updates are applied in FP32 and converted back to the
    original parameter dtype afterwards.
    """

    if "state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain a 'state_dict' key."
        )

    state_dict = checkpoint["state_dict"]

    print(
        f"Applying {len(updates)} updates."
    )

    # ---------------------------------------------------------------
    # Group SVD updates by matrix.
    #
    # Direct updates are handled separately.
    # ---------------------------------------------------------------

    svd_groups = {}
    direct_updates = []

    for update in updates:

        if update["type"] == "svd":

            svd_groups.setdefault(
                update["matrix"],
                [],
            ).append(update)

        elif update["type"] == "direct":

            direct_updates.append(update)

        else:
            raise ValueError(
                f"Unknown update type: "
                f"{update['type']}"
            )

    # ===============================================================
    # Apply SVD updates
    # ===============================================================

    for matrix_name, matrix_updates in svd_groups.items():

        if matrix_name not in state_dict:
            raise KeyError(
                f"SVD update refers to unknown matrix: "
                f"{matrix_name}"
            )

        weight = state_dict[matrix_name]

        print(
            f"\nSVD matrix: {matrix_name}"
        )

        print(
            f"  original shape: "
            f"{tuple(weight.shape)}"
        )

        # -----------------------------------------------------------
        # Reconstruct matrix representation
        #
        # 2-D:
        #     [rows, cols]
        #
        # Conv:
        #     [out, in, k, k]
        #     becomes [out, in*k*k]
        # -----------------------------------------------------------

        original_shape = weight.shape

        if weight.ndim == 2:

            matrix_shape = (
                weight.shape[0],
                weight.shape[1],
            )

        elif weight.ndim > 2:

            matrix_shape = (
                weight.shape[0],
                int(np.prod(weight.shape[1:])),
            )

        else:
            raise ValueError(
                f"SVD update applied to tensor "
                f"with invalid ndim={weight.ndim}: "
                f"{matrix_name}"
            )

        update = torch.zeros(
            matrix_shape,
            dtype=torch.float32,
        )

        # -----------------------------------------------------------
        # Apply each rank-1 component
        # -----------------------------------------------------------

        for triplet in sorted(
            matrix_updates,
            key=lambda x: x["rank"],
        ):

            rank = int(triplet["rank"])
            sigma = float(triplet["sigma"])

            u = torch.from_numpy(
                triplet["u"]
            ).to(torch.float32)

            v = torch.from_numpy(
                triplet["v"]
            ).to(torch.float32)

            if u.numel() != matrix_shape[0]:
                raise ValueError(
                    f"{matrix_name}, rank {rank}: "
                    f"U has {u.numel()} elements, "
                    f"expected {matrix_shape[0]}"
                )

            if v.numel() != matrix_shape[1]:
                raise ValueError(
                    f"{matrix_name}, rank {rank}: "
                    f"V has {v.numel()} elements, "
                    f"expected {matrix_shape[1]}"
                )

            update += (
                sigma
                * torch.outer(u, v)
            )

            print(
                f"  rank={rank:4d}, "
                f"sigma={sigma:.8e}"
            )

        # -----------------------------------------------------------
        # Restore original tensor shape
        # -----------------------------------------------------------

        if len(original_shape) > 2:

            update = update.reshape(
                original_shape
            )

        # -----------------------------------------------------------
        # Apply update
        # -----------------------------------------------------------

        original_dtype = weight.dtype

        state_dict[matrix_name] = (
            weight.detach().to(torch.float32)
            + update
        ).to(original_dtype)

    # ===============================================================
    # Apply direct 1-D updates
    # ===============================================================

    for direct in direct_updates:

        matrix_name = direct["matrix"]

        if matrix_name not in state_dict:
            raise KeyError(
                f"Direct update refers to unknown tensor: "
                f"{matrix_name}"
            )

        tensor = state_dict[matrix_name]

        delta = torch.from_numpy(
            direct["delta"]
        ).to(torch.float32)

        print(
            f"\nDirect update: {matrix_name}"
        )

        print(
            f"  shape: {tuple(tensor.shape)}"
        )

        print(
            f"  elements: {delta.numel()}"
        )

        if tensor.numel() != delta.numel():
            raise ValueError(
                f"{matrix_name}: "
                f"delta has {delta.numel()} elements, "
                f"but tensor has {tensor.numel()}."
            )

        delta = delta.reshape(
            tensor.shape
        )

        original_dtype = tensor.dtype

        state_dict[matrix_name] = (
            tensor.detach().to(torch.float32)
            + delta
        ).to(original_dtype)

    return checkpoint


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Apply binary SVD/direct model updates "
            "to a PyTorch checkpoint."
        )
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Input .pt checkpoint.",
    )

    parser.add_argument(
        "updates_path",
        type=Path,
        help="Binary SVD update file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .pt checkpoint.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------------

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: "
            f"{args.model_path}"
        )

    if not args.updates_path.exists():
        raise FileNotFoundError(
            f"Update file not found: "
            f"{args.updates_path}"
        )

    if args.output is None:
        args.output = (
            args.model_path.parent
            / f"{args.model_path.stem}_updated.pt"
        )

    print("=" * 70)
    print("Binary SVD Update Application")
    print("=" * 70)

    print(
        f"Input model : {args.model_path}"
    )

    print(
        f"Update file : {args.updates_path}"
    )

    print(
        f"Output model: {args.output}"
    )

    # ---------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        args.model_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected the .pt file to contain "
            "a dictionary checkpoint."
        )

    print(
        f"\nCheckpoint keys: "
        f"{list(checkpoint.keys())}"
    )

    # ---------------------------------------------------------------
    # Load binary updates
    # ---------------------------------------------------------------

    updates = load_binary_updates(
        args.updates_path
    )

    num_svd = sum(
        u["type"] == "svd"
        for u in updates
    )

    num_direct = sum(
        u["type"] == "direct"
        for u in updates
    )

    print(
        f"\nLoaded {len(updates)} updates:"
    )

    print(
        f"  SVD updates   : {num_svd}"
    )

    print(
        f"  Direct updates: {num_direct}"
    )

    # ---------------------------------------------------------------
    # Apply updates
    # ---------------------------------------------------------------

    checkpoint = apply_updates(
        checkpoint,
        updates,
    )

    # ---------------------------------------------------------------
    # Save updated checkpoint
    # ---------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        args.output,
    )

    print("\n" + "=" * 70)
    print(
        "Successfully saved updated checkpoint:"
    )
    print(args.output)
    print("=" * 70)


if __name__ == "__main__":
    main()