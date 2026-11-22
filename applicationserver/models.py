from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper
def flatten_conv2d(weight: torch.Tensor) -> torch.Tensor:
    """
    [C_out, C_in, K_h, K_w]
        ->
    [C_out, C_in * K_h * K_w]

    Behält die Output-Channel-Struktur.
    """
    return weight.reshape(weight.shape[0], -1)


def flatten_conv1d(weight: torch.Tensor) -> torch.Tensor:
    """
    [C_out, C_in, K]
        ->
    [C_out, C_in * K]
    """
    return weight.reshape(weight.shape[0], -1)


def flatten_linear(weight: torch.Tensor) -> torch.Tensor:
    """
    Linear weights sind bereits 2D.
    """
    return weight


def flatten_gru(weight: torch.Tensor) -> torch.Tensor:
    """
    GRU-Gewichte sind bereits 2D.

    [3 * hidden_size, input_size]
    oder
    [3 * hidden_size, hidden_size]
    """
    return weight


# Models
class FlattenableModule(nn.Module):
    """
    Interface for models that can return their weights as a list of 2D matrices.
    """

    def flatten(self) -> list[torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the flatten() method.")

class MLP_BIG(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_frames=101,
        n_classes=32,
    ):
        super().__init__()

        d = n_mels * n_frames

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def flatten(self):
        return [
            self.net[1].weight,   # [512, 4040]
            self.net[4].weight,   # [512, 512]
            self.net[7].weight,   # [32, 512]
        ]
    
class MLP(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_frames=101,
        n_classes=32
    ):
        super().__init__()

        d = n_mels * n_frames

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def flatten(self):
        """
        Trivial weight return
        """
        return [
            self.net[1].weight,  # [16, d]
            self.net[4].weight,  # [16, 16]
            self.net[6].weight,  # [classes, 16]
        ]


class CNN(FlattenableModule):
    def __init__(self, n_classes=32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x.flatten(1))

    def flatten(self):
        """
        Conv2d:
        [C_out, C_in, K, K]
        ->
        [C_out, C_in*K*K]
        """
        return [
            flatten_conv2d(self.features[0].weight),
            flatten_conv2d(self.features[4].weight),
            flatten_conv2d(self.features[8].weight),
            flatten_linear(self.classifier.weight),
        ]

# Helper Class DSConvBlock for Depthwise-Separable CNN
class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    def flatten(self):
        """
        Für Depthwise:
            [C, 1, K, K] -> [C, K*K]

        Für Pointwise:
            [C_out, C_in, 1, 1] -> [C_out, C_in]
        """
        return [
            flatten_conv2d(self.depthwise[0].weight),
            flatten_conv2d(self.pointwise[0].weight),
        ]

# Depthwise-Separable CNN
class DSCNN(FlattenableModule):
    def __init__(self, n_classes=32):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                24,
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            DSConvBlock(24, 32, 1),
            DSConvBlock(32, 48, 2),
            DSConvBlock(48, 64, 1),
            DSConvBlock(64, 64, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def flatten(self):
        matrices = []

        # Stem
        matrices.append(
            flatten_conv2d(self.stem[0].weight)
        )

        # Depthwise + pointwise blocks
        for block in self.blocks:
            matrices.extend(block.flatten())

        # Classifier
        matrices.append(self.fc.weight)

        return matrices


# GRU
class GRUModel(FlattenableModule):
    def __init__(
        self,
        n_mels=40,
        n_classes=32
    ):
        super().__init__()

        self.proj = nn.Linear(n_mels, 48)

        self.gru = nn.GRU(
            input_size=48,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(48, n_classes)

    def forward(self, x):
        # [B, mel, time] -> [B, time, mel]
        x = x.transpose(1, 2)

        x = F.relu(self.proj(x))

        _, h = self.gru(x)

        return self.fc(h[-1])

    def flatten(self):
        """
        GRU is already 2D.

        weight_ih_l0:
            [3H, input_size]

        weight_hh_l0:
            [3H, H]
        """
        return [
            self.proj.weight,
            self.gru.weight_ih_l0,
            self.gru.weight_hh_l0,
            self.fc.weight,
        ]


# CRNN

class CRNN(FlattenableModule):
    """
    Small CNN front-end followed by a single GRU layer.
    """

    def __init__(self, n_classes=32):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(12, 20, 3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.gru = nn.GRU(
            input_size=20 * 10,
            hidden_size=40,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(40, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn(x)

        b, c, m, t = x.shape

        x = (
            x.permute(0, 3, 1, 2)
             .reshape(b, t, c * m)
        )

        _, h = self.gru(x)

        return self.fc(h[-1])

    def flatten(self):
        return [
            flatten_conv2d(self.cnn[0].weight),
            flatten_conv2d(self.cnn[4].weight),

            self.gru.weight_ih_l0,
            self.gru.weight_hh_l0,

            self.fc.weight,
        ]


# CNN1D
class CNN1D(FlattenableModule):
    """
    Lightweight 1D CNN operating along the time axis.
    """

    def __init__(
        self,
        n_mels=40,
        n_classes=32
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                n_mels,
                32,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                48,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                48,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

    def flatten(self):
        """
        Conv1d:
        [C_out, C_in, K]
        ->
        [C_out, C_in*K]
        """
        return [
            flatten_conv1d(self.features[0].weight),
            flatten_conv1d(self.features[4].weight),
            flatten_conv1d(self.features[8].weight),
            self.classifier.weight,
        ]


# SVDF
class SVDF(FlattenableModule):
    """
    Lightweight SVDF network.
    """

    def __init__(
        self,
        n_mels=40,
        n_frames=101,
        rank=2,
        units=32,
        n_classes=32,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.n_frames = n_frames
        self.rank = rank
        self.units = units

        self.freq = nn.Parameter(
            torch.randn(units, rank, n_mels)
        )

        self.time = nn.Parameter(
            torch.randn(units, rank, n_frames)
        )

        self.bn = nn.BatchNorm1d(units)

        self.fc = nn.Linear(units, n_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.freq)
        nn.init.normal_(
            self.time,
            mean=0.0,
            std=0.02
        )
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = torch.einsum(
            "bmt,urm->burt",
            x,
            self.freq,
        )

        x = (
            x * self.time.unsqueeze(0)
        ).sum(dim=-1)

        x = x.sum(dim=2)

        x = self.bn(x)
        x = F.relu(x)

        return self.fc(x)

    def flatten(self):
        """
        Frequency:
            [units, rank, n_mels]
            ->
            [units * rank, n_mels]

        Temporal:
            [units, rank, n_frames]
            ->
            [units * rank, n_frames]
        """
        freq_matrix = self.freq.reshape(
            self.units * self.rank,
            self.n_mels
        )

        time_matrix = self.time.reshape(
            self.units * self.rank,
            self.n_frames
        )

        return [
            freq_matrix,
            time_matrix,
            self.fc.weight,
        ]