"""Small trainable Stage 1 separator for dialogue/music/effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Stage1SeparatorConfig:
    sample_rate: int = 32000
    chunk_seconds: float = 6.0
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    base_channels: int = 32
    num_blocks: int = 8
    targets: tuple[str, ...] = ("dialogue", "music", "effects")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["targets"] = list(self.targets)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Stage1SeparatorConfig":
        payload = dict(data)
        payload["targets"] = tuple(payload.get("targets", ("dialogue", "music", "effects")))
        return cls(**payload)


class ResidualMaskBlock(nn.Module):
    """Residual 2D block over time-frequency features."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        groups = max(1, min(8, channels // 4))
        self.norm = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return residual + x


class Stage1DialogueSeparator(nn.Module):
    """Mask-based separator producing dialogue/music/effects stems."""

    def __init__(self, config: Stage1SeparatorConfig):
        super().__init__()
        self.config = config
        self.num_targets = len(config.targets)

        self.input_proj = nn.Conv2d(1, config.base_channels, kernel_size=3, padding=1)
        dilations = [1, 2, 4, 8, 1, 2, 4, 8][: config.num_blocks]
        self.blocks = nn.ModuleList(
            ResidualMaskBlock(config.base_channels, dilation=dilation) for dilation in dilations
        )
        self.output_proj = nn.Conv2d(config.base_channels, self.num_targets, kernel_size=1)
        self.register_buffer("window", torch.hann_window(config.win_length), persistent=False)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(waveform.device),
            return_complex=True,
        )

    def _istft(self, stft_tensor: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            stft_tensor,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(stft_tensor.device),
            length=length,
        )

    def forward(self, mixture_waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if mixture_waveform.dim() == 1:
            mixture_waveform = mixture_waveform.unsqueeze(0)

        mix_stft = self._stft(mixture_waveform)
        magnitude = torch.abs(mix_stft)
        features = torch.log1p(magnitude).unsqueeze(1)

        hidden = self.input_proj(features)
        for block in self.blocks:
            hidden = block(hidden)

        mask_logits = self.output_proj(hidden)
        masks = torch.softmax(mask_logits, dim=1)
        estimate_stfts = mix_stft.unsqueeze(1) * masks

        stems = []
        for target_idx in range(self.num_targets):
            stems.append(self._istft(estimate_stfts[:, target_idx], length=mixture_waveform.shape[-1]))
        stacked_stems = torch.stack(stems, dim=1)
        return {"stems": stacked_stems, "masks": masks}


def _multi_resolution_stft_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
) -> torch.Tensor:
    batch, stems, length = prediction.shape
    pred = prediction.reshape(batch * stems, length)
    tgt = target.reshape(batch * stems, length)
    device = pred.device

    loss = pred.new_tensor(0.0)
    for fft_size in fft_sizes:
        win_length = min(fft_size, length)
        hop_length = max(64, win_length // 4)
        window = torch.hann_window(win_length, device=device)
        pred_stft = torch.stft(
            pred,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        tgt_stft = torch.stft(
            tgt,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        loss = loss + F.l1_loss(torch.abs(pred_stft), torch.abs(tgt_stft))
    return loss / len(fft_sizes)


def stage1_separator_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    waveform_weight: float = 1.0,
    stft_weight: float = 0.35,
) -> torch.Tensor:
    waveform_l1 = F.l1_loss(prediction, target)
    stft_l1 = _multi_resolution_stft_l1(prediction, target)
    return waveform_weight * waveform_l1 + stft_weight * stft_l1
