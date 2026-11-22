#!/usr/bin/env python3
"""
TinyML Speech Commands experiment.

Five small architectures:
    mlp, cnn, ds_cnn, gru, crnn

Workflow:
  1. Download Speech Commands v0.02.
  2. Build a KWS task using the dataset's available classes (core commands,
     digits, auxiliary words), plus 'unknown' and 'silence'.
  3. Train one model per architecture on an initial (small) subset of the
     training split and then incrementally expand the dataset for subsequent
     versions to simulate data collection over time.
  4. Each version is trained/finetuned sequentially from the last version and
     every resulting checkpoint is saved.

Default fine-tuning sizes (legacy): 100, 500, 1000, 5000 examples.
Use --finetune-fracs to specify relative fractions of the training split.

The script intentionally keeps the models small enough to be plausible
for a low-power/MCU-class device.

Requirements:
    pip install torch numpy scipy tqdm

Example:
    python3 train.py \
        --data-dir data/speech_commands \
        --out-dir runs \
        --device cuda

Only run the base training once. If base checkpoints already exist,
they are reused unless --force-base is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from tqdm import tqdm

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# from networkserver.models import MLP_BIG, MLP, CNN, CNN1D, DSCNN, SVDF, GRUModel, CRNN
# Models that can be trained with Low-Rank Adaptation (LoRA)
from networkserver.models_lora import MLP, CNN, DSCNN, GRUModel, CRNN, CNN1D, SVDF

MODEL_FACTORIES = {
    # "mlp_big": MLP_BIG,
    "mlp": MLP,
    "cnn": CNN,
    "cnn1d": CNN1D,
    "ds_cnn": DSCNN,
    "svdf": SVDF,
    "gru": GRUModel,
    "crnn": CRNN,
}

# Reproducibility

SEED = 1337
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000

# All target words: core commands, digits zero-nine, and auxiliary words.
TARGET_WORDS = [
    # core commands
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    # digits
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    # auxiliary words
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
]
UNKNOWN = "unknown"
SILENCE = "silence"
CLASSES = TARGET_WORDS + [UNKNOWN, SILENCE]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Dataset download / indexing
def download_dataset(data_dir: Path) -> Path:
    """
    Downloads and extracts Speech Commands v0.02 into data_dir.

    Expected result:
        data_dir/
          yes/*.wav
          no/*.wav
          ...
          background_noise/*.wav
          validation_list.txt
          testing_list.txt
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / "validation_list.txt"

    if marker.exists() and (data_dir / "testing_list.txt").exists():
        return data_dir

    archive = data_dir.parent / "speech_commands_v0.02.tar.gz"
    if not archive.exists():
        print(f"Downloading Speech Commands v0.02 to {archive} ...")
        urllib.request.urlretrieve(DATASET_URL, archive)

    print(f"Extracting {archive} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)

    # Some archives may extract directly into data_dir; others may contain
    # paths relative to the root. The official archive is flat enough that
    # the expected marker should be here.
    if not marker.exists():
        raise RuntimeError(
            f"Dataset extraction finished but {marker} was not found in {data_dir}."
        )
    return data_dir


def read_split_file(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip().replace("\\", "/") for line in f if line.strip()}


def all_wav_files(data_dir: Path) -> List[str]:
    files = []
    for p in data_dir.rglob("*.wav"):
        rel = p.relative_to(data_dir).as_posix()
        if rel.startswith("background_noise/"):
            continue
        files.append(rel)
    return files


def make_file_lists(data_dir: Path) -> Dict[str, List[str]]:
    """
    Creates train/validation/test file lists using Google's official
    validation_list.txt and testing_list.txt files.

    Returns paths relative to data_dir.
    """
    val = read_split_file(data_dir / "validation_list.txt")
    test = read_split_file(data_dir / "testing_list.txt")

    all_files = all_wav_files(data_dir)

    # Remove exact duplicates defensively.
    val = {x for x in val if (data_dir / x).exists()}
    test = {x for x in test if (data_dir / x).exists()}

    train = set(all_files) - val - test

    return {
        "train": sorted(train),
        "validation": sorted(val),
        "test": sorted(test),
    }


def label_for_path(rel_path: str) -> int:
    word = rel_path.split("/")[0]
    return CLASS_TO_ID[word] if word in CLASS_TO_ID else CLASS_TO_ID[UNKNOWN]


# Audio / features
def load_wav(path: Path) -> torch.Tensor:
    sr, audio = wavfile.read(path)

    if audio.dtype == np.int16:
        x = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        x = audio.astype(np.float32) / 2147483648.0
    else:
        x = audio.astype(np.float32)

    if x.ndim > 1:
        x = x.mean(axis=1)

    x = torch.from_numpy(x)

    if sr != SAMPLE_RATE:
        # Linear resampling is adequate for this low-rate KWS experiment.
        new_len = int(round(len(x) * SAMPLE_RATE / sr))
        x = F.interpolate(
            x.view(1, 1, -1),
            size=new_len,
            mode="linear",
            align_corners=False,
        ).view(-1)

    if len(x) < CLIP_SAMPLES:
        x = F.pad(x, (0, CLIP_SAMPLES - len(x)))
    else:
        x = x[:CLIP_SAMPLES]

    return x


def make_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 20.0,
    f_max: float = 7600.0,
) -> torch.Tensor:
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = torch.linspace(
        hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2
    )
    hz_points = torch.tensor([mel_to_hz(float(x)) for x in mel_points])

    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)

    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1:m + 2]
        left, center, right = int(left), int(center), int(right)

        if center > left:
            fb[m - 1, left:center] = torch.arange(
                left, center, dtype=torch.float32
            ) / float(center - left)

        if right > center:
            fb[m - 1, center:right] = torch.arange(
                center, right, dtype=torch.float32
            )
            fb[m - 1, center:right] = (
                right - torch.arange(center, right, dtype=torch.float32)
            ) / float(right - center)

    return fb


class FeatureExtractor:
    """
    40-bin log-Mel spectrogram.

    25 ms window / 10 ms hop gives approximately 98 time frames for a
    one-second clip. This is a common compact KWS representation.
    """

    def __init__(self, n_mels=40, n_fft=400, hop_length=160):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
        self.mel_fb = make_mel_filterbank(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=SAMPLE_RATE,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.matmul(self.mel_fb, power)
        log_mel = torch.log(mel + 1e-6)

        # Per-utterance normalization is cheap and makes training stable.
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
        return log_mel.float()  # [mel, time]


# Silence examples
def background_noise_files(data_dir: Path) -> List[Path]:
    return sorted((data_dir / "_background_noise_").glob("*.wav"))


def make_silence_audio(
    noise_files: Sequence[Path],
    index: int,
    train: bool,
) -> torch.Tensor:
    """
    Deterministically extracts a one-second segment from background noise.
    Different index ranges are used for train/validation/test.
    """
    if not noise_files:
        return torch.zeros(CLIP_SAMPLES)

    path = noise_files[index % len(noise_files)]
    x = load_wav(path)

    if len(x) < CLIP_SAMPLES:
        x = x.repeat(math.ceil(CLIP_SAMPLES / len(x)))

    # Deterministic offset; validation/test do not share the same offset rule.
    max_start = max(0, len(x) - CLIP_SAMPLES)
    salt = 17 if train else 7919
    start = (index * 7919 + salt) % (max_start + 1)
    return x[start:start + CLIP_SAMPLES]


# Dataset
class SpeechCommandsKWS(torch.utils.data.Dataset):
    """
    Speech Commands KWS dataset using the classes defined in CLASSES.

    Unknown = all non-target words.
    Silence = one-second segments from background_noise.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        feature_extractor: FeatureExtractor,
        files_by_split: Dict[str, List[str]],
        unknown_seed: int = 1337,
        max_unknown: int | None = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.fe = feature_extractor
        self.noise_files = background_noise_files(data_dir)

        rng = random.Random(unknown_seed + {"train": 1, "validation": 2, "test": 3}[split])

        files = files_by_split[split]
        target_files = [f for f in files if label_for_path(f) != CLASS_TO_ID[UNKNOWN]]
        unknown_files = [f for f in files if label_for_path(f) == CLASS_TO_ID[UNKNOWN]]

        # Keep unknown class from dominating the target words.
        # Approximately one unknown example per target-word example is enough.
        if max_unknown is None:
            max_unknown = len(target_files)

        rng.shuffle(unknown_files)
        unknown_files = unknown_files[:min(max_unknown, len(unknown_files))]

        # Keep all target examples.
        self.examples: List[Tuple[str, int]] = [
            (f, label_for_path(f)) for f in target_files
        ]
        self.examples.extend((f, CLASS_TO_ID[UNKNOWN]) for f in unknown_files)

        # Add a moderate amount of silence.
        # 10% of the non-silence examples, minimum 100.
        self.num_silence = max(100, len(self.examples) // 10)

        rng.shuffle(self.examples)

    def __len__(self):
        return len(self.examples) + self.num_silence

    def __getitem__(self, index):
        if index >= len(self.examples):
            noise_index = index - len(self.examples)
            audio = make_silence_audio(
                self.noise_files,
                noise_index,
                train=(self.split == "train"),
            )
            label = CLASS_TO_ID[SILENCE]
        else:
            rel_path, label = self.examples[index]
            audio = load_wav(self.data_dir / rel_path)

        features = self.fe(audio)
        return features, label


class BalancedSubset(torch.utils.data.Dataset):
    """
    Deterministic, class-balanced subset.

    This is what controls the amount of training data for each version. The
    requested N samples are distributed as evenly as possible over all classes
    defined in CLASSES.
    """

    def __init__(self, dataset: SpeechCommandsKWS, n_samples: int, seed: int):
        self.dataset = dataset
        self.n_samples = n_samples

        by_class = {c: [] for c in range(len(CLASSES))}
        for i in range(len(dataset)):
            if i >= len(dataset.examples):
                label = CLASS_TO_ID[SILENCE]
            else:
                label = dataset.examples[i][1]
            by_class[label].append(i)

        rng = random.Random(seed)
        selected = []

        base = n_samples // len(CLASSES)
        remainder = n_samples % len(CLASSES)

        for c in range(len(CLASSES)):
            amount = base + (1 if c < remainder else 0)
            if not by_class[c]:
                # No examples for this class in the dataset split; skip it.
                continue
            if amount > len(by_class[c]):
                # If a class does not have enough unique examples, sample with replacement
                # to allow requesting larger balanced subsets. This models collecting
                # new data later but keeps the per-class balancing deterministic.
                selected.extend(rng.choices(by_class[c], k=amount))
            else:
                selected.extend(rng.sample(by_class[c], amount))

        rng.shuffle(selected)
        self.indices = selected

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

# Training / evaluation

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()

    return correct / max(total, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += float(loss) * y.size(0)
            seen += y.size(0)

        val_acc = evaluate(model, val_loader, device)

        print(
            f"    epoch {epoch:02d}/{epochs} "
            f"loss={running_loss / max(seen,1):.4f} "
            f"val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc


def save_checkpoint(
    path: Path,
    model: nn.Module,
    architecture: str,
    metadata: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "classes": CLASSES,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def build_model(name: str) -> nn.Module:
    return MODEL_FACTORIES[name]()

@torch.no_grad()
def evaluate_detailed(model, loader, device, classes):
    model.eval()

    correct = 0
    total = 0

    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum())
        total += y.numel()

        for class_id in range(len(classes)):
            mask = y == class_id
            class_total[class_id] += int(mask.sum())
            class_correct[class_id] += int(
                (pred[mask] == y[mask]).sum()
            )

    accuracy = correct / max(total, 1)

    per_class = {}
    for class_id, class_name in enumerate(classes):
        per_class[class_name] = {
            "correct": class_correct[class_id],
            "total": class_total[class_id],
            "accuracy": (
                class_correct[class_id] / class_total[class_id]
                if class_total[class_id] > 0
                else 0.0
            ),
        }

    return accuracy, per_class


def benchmark_checkpoint(
    checkpoint_path: Path,
    val_loader,
    test_loader,
    device,
):
    print("\n" + "=" * 72)
    print("BENCHMARKING CHECKPOINT")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary containing "
            "'architecture' and 'state_dict'."
        )

    required_keys = {"architecture", "state_dict"}

    missing = required_keys - checkpoint.keys()
    if missing:
        raise KeyError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    architecture = checkpoint["architecture"]
    state_dict = checkpoint["state_dict"]

    classes = checkpoint.get("classes", CLASSES)
    metadata = checkpoint.get("metadata", {})

    print(f"Architecture: {architecture}")
    print(f"Classes:      {len(classes)}")
    print(f"Parameters:   {sum(v.numel() for v in state_dict.values() if torch.is_tensor(v)):,}")

    if metadata:
        print("\nCheckpoint metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    if architecture not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available architectures: {list(MODEL_FACTORIES.keys())}"
        )

    model = MODEL_FACTORIES[architecture]()

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load checkpoint state_dict into architecture "
            f"'{architecture}'. The checkpoint architecture and model "
            f"definition may not match.\n\n{exc}"
        ) from exc

    model.to(device)

    print("\nEvaluating validation set...")
    val_acc, val_per_class = evaluate_detailed(
        model,
        val_loader,
        device,
        classes,
    )

    print("\nEvaluating test set...")
    test_acc, test_per_class = evaluate_detailed(
        model,
        test_loader,
        device,
        classes,
    )

    print("\n" + "-" * 72)
    print("RESULTS")
    print("-" * 72)

    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    print(f"Test accuracy:       {test_acc * 100:.2f}%")

    print("\nPer-class test accuracy:")
    for class_name, result in test_per_class.items():
        print(
            f"  {class_name:>10s}: "
            f"{result['accuracy'] * 100:6.2f}% "
            f"({result['correct']}/{result['total']})"
        )

    return {
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "parameters": sum(
            v.numel()
            for v in state_dict.values()
            if torch.is_tensor(v)
        ),
        "classes": classes,
        "metadata": metadata,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "validation_per_class": val_per_class,
        "test_per_class": test_per_class,
    }

# Main experiment
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=Path, default=Path("data/speech_commands"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/speech_commands"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_FACTORIES.keys()),
        choices=list(MODEL_FACTORIES.keys()),
    )
    parser.add_argument(
        "--finetune-fracs",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 1.0],
        help=(
            "List of relative fractions of the training data to use for sequential "
            "training/finetuning (e.g. 0.1 0.3 0.5 0.7 1.0). Fractions are relative to the "
            "full training split; they are sorted and deduplicated to form the sequence "
            "of cumulative dataset sizes. Default: 0.1 0.3 0.5 0.7 1.0."
        ),
    )
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of epochs to run for each training step (each version).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force", action="store_true")

    parser.add_argument(
        "--benchmark-model",
        type=Path,
        default=None,
        help=(
            "Benchmark an existing .pt checkpoint instead of training. "
            "The checkpoint must contain 'architecture' and 'state_dict'."
        ),
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="benchmark_result.json",
        help=(
            "Benchmark output JSON file name. Default: benchmark_result.json"
        ),
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_dir = download_dataset(args.data_dir)
    files_by_split = make_file_lists(data_dir)

    print(
        f"Files: train={len(files_by_split['train'])}, "
        f"val={len(files_by_split['validation'])}, "
        f"test={len(files_by_split['test'])}"
    )

    feature_extractor = FeatureExtractor()

    # Dataset objects are shared between models so all models see exactly
    # the same examples and preprocessing.
    train_ds = SpeechCommandsKWS(
        data_dir, "train", feature_extractor, files_by_split, args.seed
    )
    val_ds = SpeechCommandsKWS(
        data_dir, "validation", feature_extractor, files_by_split, args.seed
    )
    test_ds = SpeechCommandsKWS(
        data_dir, "test", feature_extractor, files_by_split, args.seed
    )

    print(f"KWS classes: {CLASSES}")
    print(
        f"Dataset sizes: train={len(train_ds)}, "
        f"validation={len(val_ds)}, test={len(test_ds)}"
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }

    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, shuffle=False, **loader_kwargs
    )

    if args.benchmark_model is not None:
        if not args.benchmark_model.exists():
            raise FileNotFoundError(
                f"Benchmark checkpoint not found: {args.benchmark_model}"
            )

        result = benchmark_checkpoint(
            args.benchmark_model,
            val_loader,
            test_loader,
            device,
        )

        benchmark_output = args.out_dir / args.benchmark_output
        args.out_dir.mkdir(parents=True, exist_ok=True)

        with benchmark_output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("\n" + "=" * 72)
        print(f"Benchmark results saved to: {benchmark_output}")
        print("=" * 72)

        return

    summary = {}

    for model_name in args.models:
        print("\n" + "=" * 72)
        print(f"MODEL: {model_name}")
        print("=" * 72)

        model_dir = args.out_dir / model_name

        # 1. Train/load base model
        model = build_model(model_name)
        print(f"Parameters: {count_parameters(model):,}")

        # Use --finetune-fracs (fractions of the full training split) to
        # determine the sequence of cumulative dataset sizes for sequential
        # training. Fractions are sorted, deduplicated and converted to integer
        # sample counts.
        fracs = sorted(args.finetune_fracs)
        if any(f <= 0.0 or f > 1.0 for f in fracs):
            raise ValueError("--finetune-fracs values must be in (0, 1].")

        steps = [max(1, int(len(train_ds) * f)) for f in fracs]
        steps = sorted(set(min(len(train_ds), c) for c in steps))

        print(f"Using finetune fractions: {fracs}")
        print(f"Corresponding cumulative sample counts: {steps}")

        # steps is the ordered list of cumulative sample counts; the first
        # entry is trained from scratch; later entries are finetuned sequentially.

        summary[model_name] = {
            "parameters": count_parameters(model),
            "versions": {},
        }

        current_state = None
        previous_checkpoint = None

        for idx, n_samples in enumerate(steps):
            print("\n" + "=" * 40)
            print(f"Version {idx}: training with {n_samples} samples")
            print("=" * 40)

            out_path = model_dir / f"version_{idx}_{n_samples}.pt"

            # If checkpoint already exists and user did not force rerun, load and
            # continue from it without retraining.
            if out_path.exists() and not args.force:
                print(f"Found existing checkpoint {out_path}, loading...")
                ckpt = torch.load(out_path, map_location="cpu")
                current_state = ckpt["state_dict"]
                previous_checkpoint = str(out_path)

                # Evaluate loaded model on validation/test for summary
                model = build_model(model_name)
                model.load_state_dict(current_state)
                val_acc = evaluate(model, val_loader, device)
                test_acc = evaluate(model, test_loader, device)

                summary[model_name]["versions"][str(n_samples)] = {
                    "checkpoint": str(out_path),
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }

                print(f"Loaded. val={val_acc*100:.2f}% test={test_acc*100:.2f}%")
                continue

            # Prepare dataset subset and loader
            seed = args.seed + n_samples
            subset = BalancedSubset(train_ds, n_samples=n_samples, seed=seed)
            loader = torch.utils.data.DataLoader(subset, shuffle=True, **loader_kwargs)

            # Build model and load previous state if available
            model = build_model(model_name)
            if current_state is not None:
                model.load_state_dict(current_state)

            # Choose learning rate: use base_lr for the first step, finetune_lr otherwise
            lr = args.base_lr if idx == 0 else args.finetune_lr

            acc_val = train_model(
                model,
                loader,
                val_loader,
                device,
                epochs=args.epochs,
                lr=lr,
                weight_decay=args.weight_decay,
            )

            acc_test = evaluate(model, test_loader, device)

            save_checkpoint(
                out_path,
                model.cpu(),
                model_name,
                {
                    "type": "version",
                    "version_index": idx,
                    "previous_checkpoint": previous_checkpoint,
                    "seed": seed,
                    "samples": n_samples,
                    "epochs": args.epochs,
                    "learning_rate": lr,
                    "val_accuracy": acc_val,
                    "test_accuracy": acc_test,
                    "parameter_count": count_parameters(model),
                },
            )

            summary[model_name]["versions"][str(n_samples)] = {
                "checkpoint": str(out_path),
                "val_accuracy": acc_val,
                "test_accuracy": acc_test,
            }

            print(
                f"Saved: {out_path}\n"
                f"  validation accuracy: {acc_val*100:.2f}%\n"
                f"  test accuracy:       {acc_test*100:.2f}%"
            )

            # Update current_state and previous_checkpoint for next iteration
            current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            previous_checkpoint = str(out_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nExperiment finished.")
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()