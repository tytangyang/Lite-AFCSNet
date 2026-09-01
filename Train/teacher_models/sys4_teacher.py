from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .sys4_multi_device_model import MultiDeviceModelContainer
from .sys4_net import get_model


KNOWN_DEVICE_MODELS = ("a", "b", "c", "s1", "s2", "s3")
CHECKPOINT_PREFIX = "multi_device_model."


class Sys4Teacher(nn.Module):

    def __init__(
        self,
        checkpoint_path: str,
        use_device_models: bool = True,
        n_classes: int = 10,
        in_channels: int = 1,
        base_channels: int = 32,
        channels_multiplier: float = 1.8,
        expansion_rate: float = 2.1,
    ) -> None:
        super().__init__()
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

        base_model = get_model(
            n_classes=n_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            channels_multiplier=channels_multiplier,
            expansion_rate=expansion_rate,
            n_blocks=(3, 2, 1),
            strides={"b2": (1, 1), "b3": (1, 2), "b4": (2, 1)},
        )
        self.multi_device_model = MultiDeviceModelContainer(
            base_model=base_model,
            devices=KNOWN_DEVICE_MODELS,
        )
        self.use_device_models = bool(use_device_models)
        self._load_checkpoint(ckpt_path)

        # The submission checkpoint stores FP16 tensors. Loading into the freshly
        # constructed FP32 module already casts them, and this explicit call keeps
        # KL logits numerically stable during distillation.
        self.float()
        self.requires_grad_(False)
        super().train(False)

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

        source = checkpoint.get("state_dict", checkpoint)
        if not isinstance(source, dict):
            raise TypeError("Checkpoint must contain a state_dict-like mapping.")

        teacher_state = {
            key[len(CHECKPOINT_PREFIX):]: value
            for key, value in source.items()
            if key.startswith(CHECKPOINT_PREFIX)
        }
        if not teacher_state:
            examples = list(source.keys())[:10]
            raise RuntimeError(
                f"No keys starting with '{CHECKPOINT_PREFIX}' were found. "
                f"Example keys: {examples}"
            )

        # Exact architecture/checkpoint matching is required here. Do not silently
        # skip any first-place parameter.
        self.multi_device_model.load_state_dict(teacher_state, strict=True)
        print(
            f"[Sys4Teacher] exact load OK: {len(teacher_state)} tensors from "
            f"{ckpt_path}; device-aware={self.use_device_models}"
        )

    def train(self, mode: bool = True):
        # Lightning calls train()/eval() recursively. The teacher must always stay
        # frozen and in eval mode so its BatchNorm statistics never change.
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(
        self,
        log_mel: torch.Tensor,
        devices: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if self.use_device_models:
            if devices is None:
                raise ValueError("devices must be supplied when device-aware teacher routing is enabled")
            return self.multi_device_model(log_mel, devices)
        return self.multi_device_model.base_model(log_mel)
