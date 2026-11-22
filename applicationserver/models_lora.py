"""
LoRA-enabled replacement for models.py.

Based on the original model definitions supplied by the user.
Uses Microsoft's `loralib` for Linear, Conv1d and Conv2d layers.

Install:
    pip install loralib

Important:
- MLP/CNN/CNN1D use Microsoft's LoRA layers directly.
- CRNN combines Microsoft's Conv2d/Linear LoRA with a custom LoRA GRU.
- DS-CNN applies LoRA to the stem and pointwise convolutions, while
  leaving depthwise convolutions as normal frozen layers.
- SVDF keeps its original factorized representation and adds LoRA
  adapters to its frequency/time factors and classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 4
DEFAULT_LORA_DROPOUT = 0.0


def lora_linear(in_features, out_features, bias=True,
                r=DEFAULT_LORA_RANK,
                alpha=DEFAULT_LORA_ALPHA,
                dropout=DEFAULT_LORA_DROPOUT):
    return lora.Linear(
        in_features,
        out_features,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        merge_weights=False,
    )


def lora_conv2d(*args, r=DEFAULT_LORA_RANK,
                alpha=DEFAULT_LORA_ALPHA,
                dropout=DEFAULT_LORA_DROPOUT, **kwargs):
    return lora.Conv2d(
        *args,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        merge_weights=False,
        **kwargs,
    )

class LoRAConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        base_trainable=False,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 0.0

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout)
            if lora_dropout > 0
            else nn.Identity()
        )

        kernel_dim = (
            (in_channels // groups)
            * kernel_size
        )

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(r, kernel_dim)
            )

            self.lora_B = nn.Parameter(
                torch.zeros(out_channels, r)
            )

            nn.init.kaiming_uniform_(
                self.lora_A,
                a=5 ** 0.5,
            )

        # base_trainable=True is for an initial full-finetune / from-scratch
        # training run (r can be 0 or >0; either way the real conv weight
        # updates). base_trainable=False (default) is the normal LoRA setup:
        # base frozen, only lora_A/lora_B adapt.
        self.conv.weight.requires_grad = base_trainable

        if self.conv.bias is not None:
            self.conv.bias.requires_grad = base_trainable

    def forward(self, x):

        result = self.conv(x)

        if self.r > 0:
            delta_weight = (
                self.lora_B @ self.lora_A
            ).view(self.conv.weight.shape)

            delta_weight = (
                delta_weight * self.scaling
            )

            result = result + F.conv1d(
                self.lora_dropout(x),
                delta_weight,
                bias=None,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

        return result

def lora_conv1d(*args, r=DEFAULT_LORA_RANK,
                alpha=DEFAULT_LORA_ALPHA,
                dropout=DEFAULT_LORA_DROPOUT,
                base_trainable=False, **kwargs):
    return LoRAConv1d(
        *args,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        base_trainable=base_trainable,
        **kwargs,
    )


def flatten_conv2d(weight: torch.Tensor) -> torch.Tensor:
    return weight.reshape(weight.shape[0], -1)


def flatten_conv1d(weight: torch.Tensor) -> torch.Tensor:
    return weight.reshape(weight.shape[0], -1)


def flatten_linear(weight: torch.Tensor) -> torch.Tensor:
    return weight


def flatten_gru(weight: torch.Tensor) -> torch.Tensor:
    return weight


class FlattenableModule(nn.Module):
    def flatten(self) -> list[torch.Tensor]:
        raise NotImplementedError


class LoRAGRUCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        r=DEFAULT_LORA_RANK,
        alpha=DEFAULT_LORA_ALPHA,
        dropout=DEFAULT_LORA_DROPOUT,
        base_trainable=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0

        # base_trainable=True: initial full-finetune run, these update
        # normally. base_trainable=False (default): frozen base, only
        # lora_A_*/lora_B_* adapt.
        self.weight_ih = nn.Parameter(
            torch.empty(3 * hidden_size, input_size),
            requires_grad=base_trainable,
        )
        self.weight_hh = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size),
            requires_grad=base_trainable,
        )

        self.bias_ih = nn.Parameter(
            torch.zeros(3 * hidden_size),
            requires_grad=base_trainable,
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(3 * hidden_size),
            requires_grad=base_trainable,
        )

        if r > 0:
            self.lora_A_ih = nn.Parameter(
                torch.empty(r, input_size)
            )
            self.lora_B_ih = nn.Parameter(
                torch.zeros(3 * hidden_size, r)
            )

            self.lora_A_hh = nn.Parameter(
                torch.empty(r, hidden_size)
            )
            self.lora_B_hh = nn.Parameter(
                torch.zeros(3 * hidden_size, r)
            )

            nn.init.kaiming_uniform_(self.lora_A_ih, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.lora_A_hh, a=5 ** 0.5)

        self.dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)

    def effective_weight_ih(self):
        if self.r <= 0:
            return self.weight_ih

        return (
            self.weight_ih
            + self.scaling * (self.lora_B_ih @ self.lora_A_ih)
        )

    def effective_weight_hh(self):
        if self.r <= 0:
            return self.weight_hh

        return (
            self.weight_hh
            + self.scaling * (self.lora_B_hh @ self.lora_A_hh)
        )

    def forward(self, x, h):
        gi = F.linear(
            x,
            self.effective_weight_ih(),
            self.bias_ih,
        )
        gh = F.linear(
            h,
            self.effective_weight_hh(),
            self.bias_hh,
        )

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        new = torch.tanh(i_n + reset * h_n)

        h_new = (1.0 - update) * new + update * h
        return h_new


class LoRAGRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        r=DEFAULT_LORA_RANK,
        alpha=DEFAULT_LORA_ALPHA,
        dropout=DEFAULT_LORA_DROPOUT,
        base_trainable=False,
    ):
        super().__init__()

        self.cell = LoRAGRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            r=r,
            alpha=alpha,
            dropout=dropout,
            base_trainable=base_trainable,
        )

    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.shape

        if h0 is None:
            h = x.new_zeros(batch_size, self.cell.hidden_size)
        else:
            h = h0[0] if h0.dim() == 3 else h0

        outputs = []

        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        return output, h.unsqueeze(0)


class MLP_BIG(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_frames=101,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
    ):
        super().__init__()

        d = n_mels * n_frames

        self.net = nn.Sequential(
            nn.Flatten(),
            lora_linear(d, 512, r=lora_rank, alpha=lora_alpha),
            nn.ReLU(),
            nn.Dropout(0.10),
            lora_linear(512, 512, r=lora_rank, alpha=lora_alpha),
            nn.ReLU(),
            nn.Dropout(0.10),
            lora_linear(512, n_classes, r=lora_rank, alpha=lora_alpha),
        )

    def forward(self, x):
        return self.net(x)

    def flatten(self):
        return [self.net[1].weight, self.net[4].weight, self.net[7].weight]


class MLP(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_frames=101,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
    ):
        super().__init__()

        d = n_mels * n_frames

        self.net = nn.Sequential(
            nn.Flatten(),
            lora_linear(d, 16, r=lora_rank, alpha=lora_alpha),
            nn.ReLU(),
            nn.Dropout(0.10),
            lora_linear(16, 16, r=lora_rank, alpha=lora_alpha),
            nn.ReLU(),
            lora_linear(16, n_classes, r=lora_rank, alpha=lora_alpha),
        )

    def forward(self, x):
        return self.net(x)

    def flatten(self):
        return [self.net[1].weight, self.net[4].weight, self.net[6].weight]


class CNN(FlattenableModule):
    def __init__(
        self,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
    ):
        super().__init__()

        self.features = nn.Sequential(
            lora_conv2d(1, 16, 3, padding=1, r=lora_rank, alpha=lora_alpha),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            lora_conv2d(16, 24, 3, padding=1, r=lora_rank, alpha=lora_alpha),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            lora_conv2d(24, 32, 3, padding=1, r=lora_rank, alpha=lora_alpha),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = lora_linear(32, n_classes, r=lora_rank, alpha=lora_alpha)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x.flatten(1))

    def flatten(self):
        return [
            self.features[0].conv.weight,
            self.features[4].conv.weight,
            self.features[8].conv.weight,
            self.classifier.weight,
        ]


class DSConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
    ):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            lora_conv2d(in_ch, out_ch, 1, bias=False, r=lora_rank, alpha=lora_alpha),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    def flatten(self):
        return [self.depthwise[0].weight, self.pointwise[0].conv.weight]


class DSCNN(FlattenableModule):
    def __init__(
        self,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            lora_conv2d(1, 24, 3, stride=2, padding=1, bias=False, r=lora_rank, alpha=lora_alpha),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            DSConvBlock(24, 32, 1, lora_rank, lora_alpha),
            DSConvBlock(32, 48, 2, lora_rank, lora_alpha),
            DSConvBlock(48, 64, 1, lora_rank, lora_alpha),
            DSConvBlock(64, 64, 2, lora_rank, lora_alpha),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = lora_linear(64, n_classes, r=lora_rank, alpha=lora_alpha)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def flatten(self):
        matrices = [self.stem[0].conv.weight]
        for block in self.blocks:
            matrices.extend(block.flatten())
        matrices.append(self.fc.weight)
        return matrices


class GRUModel(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        base_trainable=False,
    ):
        super().__init__()

        self.proj = lora_linear(n_mels, 48, r=lora_rank, alpha=lora_alpha)

        self.gru = LoRAGRU(input_size=48, hidden_size=48, r=lora_rank, alpha=lora_alpha, base_trainable=base_trainable)

        self.fc = lora_linear(48, n_classes, r=lora_rank, alpha=lora_alpha)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.proj(x))
        _, h = self.gru(x)
        return self.fc(h[-1])

    def flatten(self):
        return [
            self.proj.weight,
            self.gru.cell.weight_ih,
            self.gru.cell.weight_hh,
            self.fc.weight,
        ]


class CRNN(FlattenableModule):
    def __init__(
        self,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        base_trainable=False,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            lora_conv2d(1, 12, 3, padding=1, r=lora_rank, alpha=lora_alpha),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            lora_conv2d(12, 20, 3, padding=1, r=lora_rank, alpha=lora_alpha),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.gru = LoRAGRU(input_size=20 * 10, hidden_size=40, r=lora_rank, alpha=lora_alpha, base_trainable=base_trainable)

        self.fc = lora_linear(40, n_classes, r=lora_rank, alpha=lora_alpha)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * m)
        _, h = self.gru(x)
        return self.fc(h[-1])

    def flatten(self):
        return [
            self.cnn[0].conv.weight,
            self.cnn[4].conv.weight,
            self.gru.cell.weight_ih,
            self.gru.cell.weight_hh,
            self.fc.weight,
        ]


class CNN1D(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        base_trainable=False,
    ):
        super().__init__()

        self.features = nn.Sequential(
            lora_conv1d(n_mels, 32, kernel_size=5, padding=2, bias=False, r=lora_rank, alpha=lora_alpha, base_trainable=base_trainable),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            lora_conv1d(32, 48, kernel_size=5, padding=2, bias=False, r=lora_rank, alpha=lora_alpha, base_trainable=base_trainable),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            lora_conv1d(48, 64, kernel_size=3, padding=1, bias=False, r=lora_rank, alpha=lora_alpha, base_trainable=base_trainable),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = lora_linear(64, n_classes, r=lora_rank, alpha=lora_alpha)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

    def flatten(self):
        return [
            self.features[0].conv.weight,
            self.features[4].conv.weight,
            self.features[8].conv.weight,
            self.classifier.weight,
        ]


class SVDF(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_frames=101,
        rank=2,
        units=32,
        n_classes=32,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        base_trainable=False,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.n_frames = n_frames
        self.rank = rank
        self.units = units

        self.freq = nn.Parameter(torch.randn(units, rank, n_mels))
        self.time = nn.Parameter(torch.randn(units, rank, n_frames))

        self.freq.requires_grad = base_trainable
        self.time.requires_grad = base_trainable

        self.freq_A = nn.Parameter(torch.empty(lora_rank, n_mels))
        self.freq_B = nn.Parameter(torch.zeros(units * rank, lora_rank))

        self.time_A = nn.Parameter(torch.empty(lora_rank, n_frames))
        self.time_B = nn.Parameter(torch.zeros(units * rank, lora_rank))

        nn.init.kaiming_uniform_(self.freq_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.time_A, a=5 ** 0.5)

        self.scaling = lora_alpha / lora_rank if lora_rank > 0 else 0.0

        self.bn = nn.BatchNorm1d(units)

        self.fc = lora_linear(units, n_classes, r=lora_rank, alpha=lora_alpha)

        nn.init.xavier_uniform_(self.freq)
        nn.init.normal_(self.time, mean=0.0, std=0.02)

    def _effective_freq(self):
        base = self.freq.reshape(self.units * self.rank, self.n_mels)
        delta = self.freq_B @ self.freq_A
        return base + self.scaling * delta

    def _effective_time(self):
        base = self.time.reshape(self.units * self.rank, self.n_frames)
        delta = self.time_B @ self.time_A
        return base + self.scaling * delta

    def forward(self, x):
        freq = self._effective_freq().reshape(self.units, self.rank, self.n_mels)
        time = self._effective_time().reshape(self.units, self.rank, self.n_frames)

        x = torch.einsum("bmt,urm->burt", x, freq)
        x = x * time.unsqueeze(0)
        x = x.sum(dim=-1)
        x = x.sum(dim=2)

        x = self.bn(x)
        x = F.relu(x)

        return self.fc(x)

    def flatten(self):
        freq_matrix = self.freq.reshape(self.units * self.rank, self.n_mels)
        time_matrix = self.time.reshape(self.units * self.rank, self.n_frames)
        return [freq_matrix, time_matrix, self.fc.weight]


def mark_only_lora_as_trainable(model, bias="none"):
    lora.mark_only_lora_as_trainable(model, bias=bias)


def lora_state_dict(model, bias="none"):
    return lora.lora_state_dict(model, bias=bias)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total else 0.0
    return total, trainable, ratio


# Architectures whose custom (non-loralib) layers hardcode a frozen base
# weight and need an explicit base_trainable=True to be fully trainable.
_SUPPORTS_BASE_TRAINABLE = {"GRU", "CRNN", "CNN1D", "SVDF"}


def build_model(name, lora_rank=DEFAULT_LORA_RANK, lora_alpha=DEFAULT_LORA_ALPHA,
                 full_finetune=False, **kwargs):
    """
    full_finetune=False (default): normal LoRA setup. Base weights of
        loralib layers (MLP/MLP_BIG/CNN/DSCNN's lora.Linear/lora.Conv2d)
        are frozen automatically whenever lora_rank>0; everything else is
        frozen by mark_only_lora_as_trainable(). Only lora_* adapters train.

    full_finetune=True: for training an initial ("version 0") checkpoint
        from scratch or continuing to fully fine-tune. All base weights
        train, including the ones in GRU/CRNN/CNN1D/SVDF that would
        otherwise be hardcoded frozen. mark_only_lora_as_trainable is
        skipped, so biases and BatchNorm affine params train too. Save
        this checkpoint's state_dict, then build the model again with
        full_finetune=False and load_state_dict(...) from it to continue
        as a normal LoRA-adapter-only run.

        Tip: pass lora_rank=0 for MLP/MLP_BIG/CNN/DSCNN during this phase
        to skip the (unused) LoRA math entirely -- their loralib layers
        only freeze the base weight when r>0, so r=0 already gives full
        trainability with no extra parameters carried along.
    """
    models = {
        "MLP": MLP,
        "MLP_BIG": MLP_BIG,
        "CNN": CNN,
        "CNN1D": CNN1D,
        "DSCNN": DSCNN,
        "GRU": GRUModel,
        "CRNN": CRNN,
        "SVDF": SVDF,
    }

    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {list(models.keys())}")

    if full_finetune and name in _SUPPORTS_BASE_TRAINABLE:
        kwargs["base_trainable"] = True

    model = models[name](lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)

    if not full_finetune:
        mark_only_lora_as_trainable(model)

    return model