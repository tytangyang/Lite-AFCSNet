"""DCASE 2025 Task 1 inference API for the V4 LC-AFF multi-device model."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional, Sequence, Tuple
import importlib.resources as pkg_resources

import torch
import torchaudio
from torch import Tensor
from tqdm import tqdm

from .models.net import get_two_stage_model
from .models.multi_device_model import MultiDeviceModelContainer
from . import ckpts


CLASS_ORDER = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
]


@dataclass(frozen=True)
class Config:
    """Audio preprocessing and V4 model configuration used during training."""

    # Audio / log-Mel parameters
    sample_rate: int = 32000
    window_length: int = 3072
    hop_length: int = 500
    n_fft: int = 4096
    n_mels: int = 256
    f_min: int = 0
    f_max: Optional[int] = None

    # V4 student architecture
    n_classes: int = 10
    in_channels: int = 1
    base_channels: int = 32
    channels_multiplier: float = 1.8
    expansion_rate: float = 2.1
    n_blocks: Tuple[int, int] = (3, 2)
    fuse_type: str = "AFF"
    conv_type: str = "shuffle_pw"
    pw_groups: int = 4
    group_conv_groups: int = 4
    sparse_ratio: float = 0.5
    use_asym_dw: bool = True
    dw_activation: str = "silu"
    use_maxavg_stage_fusion: bool = True
    asym_dw_kernel_size: Tuple[int, int] = (3, 3)
    large_asym_dw_kernel_size: Tuple[int, int] = (5, 3)
    large_kernel_blocks: Tuple[str, str] = ("b4", "b5")

    # Devices fine-tuned by train_device_specific.py
    device_ids: Tuple[str, ...] = ("a", "b", "c", "s1", "s2", "s3")


class Baseline(torch.nn.Module):
    """Inference wrapper containing log-Mel preprocessing and device routing."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.class_order = list(CLASS_ORDER)

        # Resampling is performed by load_inputs because input files can have
        # different original sample rates.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )

        base_model = get_two_stage_model(
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channels_multiplier=config.channels_multiplier,
            expansion_rate=config.expansion_rate,
            n_blocks=config.n_blocks,
            strides={"b2": (1, 1), "b3": (1, 2), "b4": (2, 1)},
            fuse_type=config.fuse_type,
            conv_type=config.conv_type,
            pw_groups=config.pw_groups,
            group_conv_groups=config.group_conv_groups,
            sparse_ratio=config.sparse_ratio,
            use_asym_dw=config.use_asym_dw,
            dw_activation=config.dw_activation,
            use_maxavg_stage_fusion=config.use_maxavg_stage_fusion,
            asym_dw_kernel_size=config.asym_dw_kernel_size,
            large_asym_dw_kernel_size=config.large_asym_dw_kernel_size,
            large_kernel_blocks=config.large_kernel_blocks,
        )
        self.model = MultiDeviceModelContainer(base_model, list(config.device_ids))
        self.model.eval()

    def preprocess(self, waveform: Tensor) -> Tensor:
        """Convert `[B, 1, samples]` float waveforms to log-Mel features."""
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(
                "waveform must have shape [batch, 1, samples], "
                f"got {tuple(waveform.shape)}"
            )
        mel = self.mel(waveform.float())
        log_mel = (mel + 1e-5).log()
        model_dtype = next(self.model.parameters()).dtype
        return log_mel.to(dtype=model_dtype)

    def forward(self, waveform: Tensor, device_ids: Sequence[str]) -> Tensor:
        if waveform.shape[0] != len(device_ids):
            raise ValueError("Batch size and number of device IDs must match.")
        normalized_ids = [_normalize_device_id(device_id) for device_id in device_ids]
        with torch.inference_mode():
            features = self.preprocess(waveform)
            return self.model(features, normalized_ids)


def _normalize_device_id(device_id: object) -> str:
    """Normalize pandas/NumPy/string device IDs without changing routing rules."""
    if isinstance(device_id, bytes):
        device_id = device_id.decode("utf-8")
    return str(device_id).strip().lower()


def _safe_torch_load(path: str):
    """Load tensor checkpoints across old and new PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location="cpu")
    except Exception as weights_only_error:
        # Lightning checkpoints created by older versions can contain harmless
        # metadata that is rejected by weights_only=True. Only load trusted
        # checkpoints with this fallback.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raise weights_only_error


def _extract_state_dict(checkpoint) -> MutableMapping[str, Tensor]:
    """Extract a tensor state dict from common raw and Lightning formats."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint)!r}.")

    candidate = checkpoint
    for field in ("state_dict", "model_state_dict"):
        value = checkpoint.get(field)
        if isinstance(value, Mapping):
            candidate = value
            break

    state_dict = {
        str(key): value
        for key, value in candidate.items()
        if isinstance(value, (Tensor, torch.nn.Parameter))
    }
    if not state_dict:
        raise ValueError("No tensor state_dict was found in the checkpoint.")
    return state_dict


def _strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _load_checkpoint_weights(model: Baseline, checkpoint) -> str:
    """
    Load either a full multi-device checkpoint or a base-only checkpoint.

    Supported examples:
      * Lightning: multi_device_model.base_model.* / device_models.*
      * Raw container: base_model.* / device_models.*
      * Whole Baseline: model.base_model.* / model.device_models.*
      * Base Lightning: model.in_c.*
      * Raw base model: in_c.*
    """
    raw_state = _extract_state_dict(checkpoint)

    # First try to construct the MultiDeviceModelContainer state dict.
    container_state = {}
    for original_key, value in raw_state.items():
        key = _strip_module_prefix(original_key)

        for prefix in ("multi_device_model.", "model.model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break

        if key.startswith("model."):
            remainder = key[len("model.") :]
            if remainder.startswith(("base_model.", "device_models.")):
                key = remainder

        if key.startswith(("base_model.", "device_models.")):
            container_state[key] = value

    if container_state:
        expected = set(model.model.state_dict().keys())
        found = set(container_state.keys())
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        if missing or unexpected:
            raise RuntimeError(
                "The multi-device checkpoint does not match the V4 inference model. "
                f"Missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}; "
                f"unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}."
            )
        model.model.load_state_dict(container_state, strict=True)
        return "multi-device"

    # Otherwise interpret it as a single/base-model checkpoint and copy the
    # loaded base weights into all known device branches.
    expected_base = set(model.model.base_model.state_dict().keys())
    base_state = {}
    for original_key, value in raw_state.items():
        key = _strip_module_prefix(original_key)
        for prefix in (
            "multi_device_model.base_model.",
            "model.base_model.",
            "base_model.",
            "model.",
        ):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key in expected_base:
            base_state[key] = value

    missing = sorted(expected_base - set(base_state.keys()))
    unexpected = sorted(set(base_state.keys()) - expected_base)
    if missing or unexpected:
        sample_keys = list(raw_state.keys())[:8]
        raise RuntimeError(
            "Could not map the checkpoint to either the multi-device container "
            "or the V4 base model. Verify that Config matches the training run. "
            f"Missing base keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}; "
            f"checkpoint key examples: {sample_keys}."
        )

    model.model.base_model.load_state_dict(base_state, strict=True)
    for device_model in model.model.device_models.values():
        device_model.load_state_dict(base_state, strict=True)
    return "base-only"


def _default_checkpoint_path() -> str:
    resource = pkg_resources.files(ckpts).joinpath("Lite_AFCSNet.ckpt")
    if not resource.is_file():
        raise FileNotFoundError(
            "Packaged checkpoint not found. Copy the final device-specific "
            "Lightning checkpoint to DACES_T1/ckpts/Lite_AFCSNet.ckpt, or pass "
            "model_file_path explicitly."
        )
    # A Traversable can require temporary extraction when installed in a zip.
    # The caller keeps the context alive while loading via _load_default_model.
    return str(resource)


def load_model(model_file_path: Optional[str] = None) -> Baseline:
    """Build the V4 model and load packaged or explicitly supplied weights."""
    model = Baseline(Config())

    if model_file_path is None:
        resource = pkg_resources.files(ckpts).joinpath("Lite_AFCSNet.ckpt")
        if not resource.is_file():
            raise FileNotFoundError(
                "DACES_T1/ckpts/Lite_AFCSNet.ckpt is missing. Copy the final "
                "train_device_specific.py checkpoint there."
            )
        with pkg_resources.as_file(resource) as checkpoint_path:
            checkpoint = _safe_torch_load(str(checkpoint_path))
    else:
        checkpoint_path = Path(model_file_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = _safe_torch_load(str(checkpoint_path))

    model.checkpoint_type = _load_checkpoint_weights(model, checkpoint)
    model.model.eval()

    # The official complexity script counts the returned submodel. Keeping
    # network weights in FP16 matches the intended submitted parameter memory.
    # The Mel transform stays FP32; predict() switches the network to FP32 on
    # CPU because FP16 convolution is extremely slow on many CPUs.
    model.model.half()
    return model


def load_inputs(
    file_paths: List[str],
    device_ids: List[str],
    model: Baseline,
    num_workers: int = 16,
    batch_size: int = 256,
) -> List[Tensor]:
    """
    Load, mono-mix, resample, and preprocess audio files.

    Returns one tensor per file with shape `[1, 1, n_mels, frames]`, preserving
    input order. Files are grouped by both sample rate and length so zero
    padding never changes a sample's log-Mel representation.
    """
    if len(file_paths) != len(device_ids):
        raise ValueError("Number of file paths and device IDs must match.")
    if not file_paths:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model_device = next(model.model.parameters()).device
    target_sr = model.config.sample_rate

    def _load(index_and_path):
        index, path = index_and_path
        path = str(path)
        if not Path(path).is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform, sample_rate = torchaudio.load(path)
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {path}")
        waveform = waveform.float().mean(dim=0)  # [samples]
        return index, waveform, int(sample_rate)

    indexed_paths = list(enumerate(file_paths))
    worker_count = max(1, min(int(num_workers), len(indexed_paths)))
    if worker_count == 1:
        loaded = [_load(item) for item in tqdm(indexed_paths, desc="Loading files")]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded = list(
                tqdm(
                    executor.map(_load, indexed_paths),
                    total=len(indexed_paths),
                    desc="Loading files",
                )
            )

    groups = defaultdict(list)
    for index, waveform, sample_rate in loaded:
        groups[(sample_rate, waveform.numel())].append((index, waveform))

    outputs: List[Optional[Tensor]] = [None] * len(file_paths)
    for (sample_rate, _), items in tqdm(groups.items(), desc="Log-Mel preprocessing"):
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            indices, waveforms = zip(*chunk)
            batch_waveform = torch.stack(waveforms, dim=0).unsqueeze(1).to(model_device)

            if sample_rate != target_sr:
                batch_waveform = torchaudio.functional.resample(
                    batch_waveform,
                    orig_freq=sample_rate,
                    new_freq=target_sr,
                )

            with torch.inference_mode():
                batch_features = model.preprocess(batch_waveform)

            for index, features in zip(indices, batch_features):
                outputs[index] = features.unsqueeze(0).cpu()

    if any(item is None for item in outputs):
        raise RuntimeError("Internal error: one or more audio inputs were not processed.")
    return outputs  # type: ignore[return-value]


def get_model_for_device(model: Baseline, device_id: str) -> torch.nn.Module:
    """Return a known device branch, or the base model for unseen devices."""
    submodel = model.model.get_model_for_device(_normalize_device_id(device_id))
    submodel.eval()
    return submodel


def predict(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64,
) -> Tuple[List[Tensor], List[str]]:
    """Return logits in input order and the fixed ten-class label order."""
    if len(file_paths) != len(device_ids):
        raise ValueError("Number of file paths and device IDs must match.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not file_paths:
        return [], list(CLASS_ORDER)

    run_device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    model = load_model(model_file_path).to(run_device).eval()

    if run_device.type == "cuda":
        model.model.half()
    else:
        # CPU FP16 is very slow or unsupported for several convolution kernels.
        # Complexity remains measured in FP16 because load_model() returns FP16.
        model.model.float()

    inputs = load_inputs(file_paths, device_ids, model)

    # Group by device and frame count. This avoids padding log-Mel tensors and
    # guarantees that unseen devices use the base model as a single batch.
    grouped = defaultdict(list)
    normalized_ids = [_normalize_device_id(device_id) for device_id in device_ids]
    for index, (features, device_id) in enumerate(zip(inputs, normalized_ids)):
        grouped[(device_id, int(features.shape[-1]))].append((index, features))

    outputs: List[Optional[Tensor]] = [None] * len(inputs)
    for (device_id, _), items in tqdm(grouped.items(), desc="Batched inference"):
        submodel = get_model_for_device(model, device_id)
        submodel_dtype = next(submodel.parameters()).dtype

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            indices, tensors = zip(*chunk)
            batch = torch.cat(tensors, dim=0).to(
                device=run_device,
                dtype=submodel_dtype,
                non_blocking=run_device.type == "cuda",
            )

            with torch.inference_mode():
                logits = submodel(batch).float().cpu()

            for index, logit in zip(indices, logits):
                outputs[index] = logit

    if any(item is None for item in outputs):
        raise RuntimeError("Internal error: one or more predictions are missing.")
    return outputs, list(model.class_order)  # type: ignore[return-value]
