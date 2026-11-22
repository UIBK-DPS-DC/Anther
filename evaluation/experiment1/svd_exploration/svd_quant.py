import os
import struct
import torch
import numpy as np

from models import (
    FlattenableModule,
    MLP,
    CNN,
    CNN1D,
    DSCNN,
    SVDF,
    GRUModel,
    CRNN,
)


# ============================================================
# Model registry
# ============================================================

MODEL_REGISTRY = {
    "mlp": MLP,
    "cnn": CNN,
    "cnn1d": CNN1D,
    "ds_cnn": DSCNN,
    "svdf": SVDF,
    "gru": GRUModel,
    "crnn": CRNN,
}


DEFAULT_MODEL_KWARGS = {
    "mlp": {
        "n_mels": 40,
        "n_frames": 101,
        "n_classes": 32,
    },
    "cnn": {
        "n_classes": 32,
    },
    "cnn1d": {
        "n_mels": 40,
        "n_classes": 32,
    },
    "ds_cnn": {
        "n_classes": 32,
    },
    "svdf": {
        "n_mels": 40,
        "n_frames": 101,
        "rank": 2,
        "units": 32,
        "n_classes": 32,
    },
    "gru": {
        "n_mels": 40,
        "n_classes": 32,
    },
    "crnn": {
        "n_classes": 32,
    },
}


# ============================================================
# Update types
# ============================================================

TYPE_SVD = 0
TYPE_DIRECT = 1


# ============================================================
# Quantization helpers
# ============================================================

def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8):
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


# ============================================================
# Serialization
# ============================================================

def serialize_svd_triplets(
    triplets: list[dict],
    compression_level: int = 9,
) -> bytes:
    """
    Serialize quantized SVD + direct updates.

    Format before compression:

        Header:
            4 bytes  magic = b"SVDQ"
            1 byte   format version
            2 bytes  number of updates

        Per update:
            2 bytes  matrix-name length
            N bytes  matrix name
            1 byte   update type

        SVD:
            2 bytes  rank
            2 bytes  U length
            2 bytes  V length

            4 bytes  sigma FP32

            4 bytes  U scale FP32
            4 bytes  U zero_point INT32
            U_len    U UINT8

            4 bytes  V scale FP32
            4 bytes  V zero_point INT32
            V_len    V UINT8

        Direct:
            2 bytes  number of elements
            4 bytes  delta scale FP32
            4 bytes  delta zero_point INT32
            N bytes  delta UINT8

    The complete payload is compressed using zlib.

    Reconstruction of SVD update:

        Uf = (Uq - U_zero_point) * U_scale
        Vf = (Vq - V_zero_point) * V_scale

        delta ~= sigma * Uf * Vf^T

    Reconstruction of direct update:

        delta ~= (delta_q - delta_zero_point) * delta_scale
    """

    buf = bytearray()

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    buf.extend(b"SVDQ")

    # Format version.
    #
    # Version 2: values are asymmetrically (uint8 + zero_point)
    # quantized, rather than the version-1 symmetric int8 scheme.
    buf.extend(
        struct.pack(
            "<B",
            2,
        )
    )

    # Number of updates.
    if len(triplets) > 65535:
        raise ValueError(
            "Too many updates for the binary format."
        )

    buf.extend(
        struct.pack(
            "<H",
            len(triplets),
        )
    )

    # ------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------

    for triplet in triplets:

        matrix = triplet["matrix"]

        matrix_bytes = matrix.encode(
            "utf-8"
        )

        if len(matrix_bytes) > 65535:
            raise ValueError(
                f"Matrix name too long: {matrix}"
            )

        # Matrix name length.
        buf.extend(
            struct.pack(
                "<H",
                len(matrix_bytes),
            )
        )

        # Matrix name.
        buf.extend(
            matrix_bytes
        )

        update_type = triplet.get(
            "type",
            "svd",
        )

        # ========================================================
        # SVD UPDATE
        # ========================================================

        if update_type == "svd":

            buf.extend(
                struct.pack(
                    "<B",
                    TYPE_SVD,
                )
            )

            rank = int(
                triplet["rank"]
            )

            sigma = float(
                triplet["sigma"]
            )

            # ----------------------------------------------------
            # Quantize U
            # ----------------------------------------------------

            u_quantized, u_scale, u_zero_point = (
                quantize_tensor(
                    triplet["u"]
                )
            )

            # ----------------------------------------------------
            # Quantize V
            # ----------------------------------------------------

            v_quantized, v_scale, v_zero_point = (
                quantize_tensor(
                    triplet["v"]
                )
            )

            if len(u_quantized) > 65535:
                raise ValueError(
                    f"U vector too large for {matrix}"
                )

            if len(v_quantized) > 65535:
                raise ValueError(
                    f"V vector too large for {matrix}"
                )

            # Rank.
            buf.extend(
                struct.pack(
                    "<H",
                    rank,
                )
            )

            # U/V dimensions.
            buf.extend(
                struct.pack(
                    "<HH",
                    len(u_quantized),
                    len(v_quantized),
                )
            )

            # Sigma remains FP32.
            buf.extend(
                struct.pack(
                    "<f",
                    sigma,
                )
            )

            # ----------------------------------------------------
            # U
            # ----------------------------------------------------

            buf.extend(
                struct.pack(
                    "<f",
                    u_scale,
                )
            )

            buf.extend(
                struct.pack(
                    "<i",
                    u_zero_point,
                )
            )

            buf.extend(
                u_quantized.numpy().tobytes()
            )

            # ----------------------------------------------------
            # V
            # ----------------------------------------------------

            buf.extend(
                struct.pack(
                    "<f",
                    v_scale,
                )
            )

            buf.extend(
                struct.pack(
                    "<i",
                    v_zero_point,
                )
            )

            buf.extend(
                v_quantized.numpy().tobytes()
            )

        # ========================================================
        # DIRECT UPDATE
        # ========================================================

        elif update_type == "direct":

            buf.extend(
                struct.pack(
                    "<B",
                    TYPE_DIRECT,
                )
            )

            delta_quantized, delta_scale, delta_zero_point = (
                quantize_tensor(
                    triplet["delta"]
                )
            )

            if len(delta_quantized) > 65535:
                raise ValueError(
                    f"Direct update too large for {matrix}"
                )

            # Number of elements.
            buf.extend(
                struct.pack(
                    "<H",
                    len(delta_quantized),
                )
            )

            # Quantization scale.
            buf.extend(
                struct.pack(
                    "<f",
                    delta_scale,
                )
            )

            # Quantization zero point.
            buf.extend(
                struct.pack(
                    "<i",
                    delta_zero_point,
                )
            )

            # UINT8 delta.
            buf.extend(
                delta_quantized.numpy().tobytes()
            )

        else:

            raise ValueError(
                f"Unknown update type: "
                f"{update_type!r}"
            )

    return bytes(buf)


# ============================================================
# SVD + Quantization Decomposer
# ============================================================

class SVDQuantizedDeltaDecomposer:
    """
    Converts the difference between two model versions into
    rank-1 SVD updates and quantizes the SVD factors to 8 bits
    (asymmetric, per-vector affine quantization).

    Processing:

        old model
              |
              v
        new model - old model
              |
              v
        FP32 delta
              |
              v
        SVD decomposition
              |
              v
        select ranks according to svd_information
              |
              v
        quantize U and V to UINT8 (scale + zero_point)
              |
              v
        compressed binary update

    SVD update:

        delta ~= sigma * U * V^T

    Stored representation:

        Uq, U_scale, U_zero_point
        Vq, V_scale, V_zero_point
        sigma

    where:

        U ~= (Uq - U_zero_point) * U_scale
        V ~= (Vq - V_zero_point) * V_scale
    """

    def decompose(
        self,
        old_model_path: str,
        new_model_path: str,
        svd_information: float,
        model_kwargs: dict | None = None,
    ):

        if not 0.0 < svd_information <= 1.0:
            raise ValueError(
                "svd_information must be > 0 and <= 1."
            )

        # --------------------------------------------------------
        # Load checkpoints
        # --------------------------------------------------------

        old_checkpoint = torch.load(
            old_model_path,
            map_location="cpu",
        )

        new_checkpoint = torch.load(
            new_model_path,
            map_location="cpu",
        )

        old_state = (
            old_checkpoint["state_dict"]
        )

        new_state = (
            new_checkpoint["state_dict"]
        )

        all_updates = []

        # ========================================================
        # Process every tensor
        # ========================================================

        for key in old_state:

            old_tensor = old_state[key]
            new_tensor = new_state[key]

            if old_tensor.shape != new_tensor.shape:

                raise ValueError(
                    f"{key}: "
                    f"old={old_tensor.shape}, "
                    f"new={new_tensor.shape}"
                )

            # ----------------------------------------------------
            # Skip scalar buffers
            # ----------------------------------------------------

            original_shape = tuple(
                old_tensor.shape
            )

            if old_tensor.ndim == 0:

                continue

            # ====================================================
            # 1D tensors
            #
            # Keep these as direct quantized updates.
            # ====================================================

            if old_tensor.ndim == 1:

                delta = (
                    new_tensor.float()
                    - old_tensor.float()
                )

                if torch.all(delta == 0):

                    print(
                        f"  Matrix {key}: "
                        f"zero delta, skipping"
                    )

                    continue

                all_updates.append(
                    {
                        "matrix": key,
                        "type": "direct",
                        "delta": delta.clone(),
                    }
                )

                print(
                    f"  Matrix {key}: "
                    f"direct quantized update, "
                    f"elements={delta.numel()}"
                )

                continue

            # ====================================================
            # Construct FP32 delta matrix
            # ====================================================

            if old_tensor.ndim > 2:

                delta = (
                    new_tensor.float()
                    - old_tensor.float()
                ).reshape(
                    old_tensor.shape[0],
                    -1,
                ).numpy()

            else:

                delta = (
                    new_tensor.float()
                    - old_tensor.float()
                ).numpy()

            # ----------------------------------------------------
            # SVD
            # ----------------------------------------------------

            U, S, Vh = np.linalg.svd(
                delta,
                full_matrices=False,
            )

            # ----------------------------------------------------
            # Total information
            # ----------------------------------------------------

            total_information = float(
                np.sum(S ** 2)
            )

            if total_information == 0:

                print(
                    f"  Matrix {key}: "
                    f"zero delta, skipping"
                )

                continue

            target_information = (
                svd_information
                * total_information
            )

            # ----------------------------------------------------
            # Select SVD ranks
            # ----------------------------------------------------

            cumulative_information = 0.0

            selected_updates = []

            for rank, sigma in enumerate(S):

                cumulative_information += (
                    float(sigma ** 2)
                )

                selected_updates.append(
                    {
                        "matrix": key,
                        "type": "svd",
                        "rank": rank,
                        "sigma": float(sigma),
                        "shape": original_shape,
                        "u": torch.from_numpy(
                            U[:, rank].copy()
                        ),
                        "v": torch.from_numpy(
                            Vh[rank, :].copy()
                        ),
                    }
                )

                if (
                    cumulative_information
                    >= target_information
                ):
                    break

            actual_information = (
                cumulative_information
                / total_information
            )

            # ====================================================
            # Print information
            # ====================================================

            print(
                f"\n  Matrix: {key}"
            )

            print(
                f"    original shape: "
                f"{original_shape}"
            )

            print(
                f"    total rank: "
                f"{len(S)}"
            )

            print(
                f"    selected rank: "
                f"{len(selected_updates)}"
            )

            print(
                f"    requested information: "
                f"{svd_information:.4f}"
            )

            print(
                f"    actual information: "
                f"{actual_information:.4f}"
            )

            print(
                f"    U shape: "
                f"{U.shape}"
            )

            print(
                f"    Vh shape: "
                f"{Vh.shape}"
            )

            all_updates.extend(
                selected_updates
            )

        return all_updates


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Generate quantized SVD model updates."
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
        "--svd_information",
        type=float,
        default=1.0,
        help=(
            "Fraction of SVD information to retain "
            "(default: 1.0)"
        ),
    )

    parser.add_argument(
        "--output",
        default="svd_quant_update.bin",
        help=(
            "Output binary file "
            "(default: svd_quant_update.bin)"
        ),
    )

    args = parser.parse_args()

    print(
        f"Loading old model from: "
        f"{args.old}"
    )

    print(
        f"Loading new model from: "
        f"{args.new}"
    )

    print(
        f"SVD information: "
        f"{args.svd_information}"
    )

    decomposer = (
        SVDQuantizedDeltaDecomposer()
    )

    try:

        updates = decomposer.decompose(
            args.old,
            args.new,
            args.svd_information,
        )

    except Exception as exc:

        print(
            f"Failed to compute "
            f"SVD + quantized delta: {exc}"
        )

        sys.exit(2)

    print(
        f"\nTotal updates generated: "
        f"{len(updates)}"
    )

    if not updates:

        print(
            "No differences found."
        )

        sys.exit(0)

    # ------------------------------------------------------------
    # Print updates
    # ------------------------------------------------------------

    svd_count = 0
    direct_count = 0

    for update in updates:

        if update.get("type") == "direct":

            delta = update["delta"]

            direct_count += 1

            print(
                f"matrix={update['matrix']}, "
                f"type=direct-uint8, "
                f"elements={delta.numel()}"
            )

        else:

            u = update["u"]
            v = update["v"]

            svd_count += 1

            print(
                f"matrix={update['matrix']}, "
                f"type=svd-uint8, "
                f"rank={update['rank']:3d}, "
                f"sigma={update['sigma']:.6e}, "
                f"u={len(u)}, "
                f"v={len(v)}"
            )

    print()
    print(
        f"SVD updates:    {svd_count}"
    )

    print(
        f"Direct updates: {direct_count}"
    )

    # ------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------

    serialized_data = (
        serialize_svd_triplets(
            updates
        )
    )

    with open(
        args.output,
        "wb",
    ) as f:

        f.write(
            serialized_data
        )

    print(
        f"\nSaved "
        f"{len(updates)} quantized SVD updates "
        f"to: {args.output}"
    )

    print(
        f"Output size: "
        f"{len(serialized_data) / 1024:.2f} KiB"
    )