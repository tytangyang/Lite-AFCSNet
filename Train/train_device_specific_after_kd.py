import argparse
import copy
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import transformers
import wandb

try:
    from dataset.dcase25 import get_training_set, get_test_set
except ModuleNotFoundError:
    from dataset25.dcase25 import get_training_set, get_test_set

from helpers.init import worker_init_fn
from helpers.utils import probabilistic
from helpers import complexity
from models.net import get_two_stage_model
from models.multi_device_model import MultiDeviceModelContainer


TRAIN_DEVICES = ["a", "b", "c", "s1", "s2", "s3"]
ALL_DEVICES = ["a", "b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]
LABEL_NAMES = [
    "airport", "bus", "metro", "metro_station", "park",
    "public_square", "shopping_mall", "street_pedestrian",
    "street_traffic", "tram",
]
DEVICE_GROUPS = {
    "a": "real", "b": "real", "c": "real",
    "s1": "seen", "s2": "seen", "s3": "seen",
    "s4": "unseen", "s5": "unseen", "s6": "unseen",
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class AdaptiveSpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        max_num_masks: int,
        schedule: str,
        total_epochs: int,
    ):
        super().__init__()
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.max_num_masks = int(max_num_masks)
        self.schedule = schedule
        self.total_epochs = int(total_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_num_masks <= 0:
            return x
        if self.schedule == "linear":
            progress = min(1.0, (self.current_epoch + 1) / max(1, self.total_epochs))
            max_masks_now = max(1, round(self.max_num_masks * progress))
        elif self.schedule == "step":
            max_masks_now = 1 if self.current_epoch < self.total_epochs // 2 else self.max_num_masks
        else:
            max_masks_now = self.max_num_masks

        n_freq = int(torch.randint(0, max_masks_now + 1, (1,)).item())
        n_time = int(torch.randint(0, max_masks_now + 1, (1,)).item())
        for _ in range(n_freq):
            x = T.FrequencyMasking(self.freq_mask_param, iid_masks=True)(x)
        for _ in range(n_time):
            x = T.TimeMasking(self.time_mask_param, iid_masks=True)(x)
        return x


def build_student(config) -> nn.Module:
    return get_two_stage_model(
        n_classes=config.n_classes,
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        channels_multiplier=config.channels_multiplier,
        expansion_rate=config.expansion_rate,
        fuse_type=config.fuse_type,
        conv_type=config.conv_type,
        pw_groups=config.pw_groups,
        group_conv_groups=config.group_conv_groups,
        sparse_ratio=config.sparse_ratio,
        use_asym_dw=config.use_asym_dw,
        dw_activation=config.dw_activation,
        use_maxavg_stage_fusion=config.use_maxavg_stage_fusion,
        asym_dw_kernel_size=(config.asym_dw_freq, config.asym_dw_time),
        large_asym_dw_kernel_size=(config.large_asym_dw_freq, config.large_asym_dw_time),
        large_kernel_blocks=tuple(
            item.strip() for item in config.large_kernel_blocks.split(",") if item.strip()
        ),
    )


def extract_student_state_dict(checkpoint_path: str, model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the distilled general student's model weights.

    Supports Lightning checkpoints from train_base_kd_sys4.py.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Distilled student checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}")

    target = model.state_dict()
    prefixes = ["model.", "student.", "student_model.", "module.model.", ""]
    best_prefix = None
    best_state = None
    best_score = -1

    for prefix in prefixes:
        candidate = {}
        for key, value in source.items():
            if prefix and not key.startswith(prefix):
                continue
            stripped = key[len(prefix):] if prefix else key
            if stripped in target and tuple(value.shape) == tuple(target[stripped].shape):
                candidate[stripped] = value
        if len(candidate) > best_score:
            best_prefix = prefix
            best_state = candidate
            best_score = len(candidate)

    assert best_state is not None
    missing = sorted(set(target) - set(best_state))
    if missing:
        preview = "\n  ".join(missing[:20])
        raise RuntimeError(
            "The KD checkpoint does not exactly match the current tiny-student architecture.\n"
            f"Selected prefix: {best_prefix!r}\n"
            f"Loaded tensors: {len(best_state)}/{len(target)}\n"
            f"First missing keys:\n  {preview}\n"
            "Use exactly the same architecture arguments as the KD run."
        )

    print(
        f"[KD checkpoint] loaded general student from {path}\n"
        f"[KD checkpoint] selected prefix={best_prefix!r}; tensors={len(best_state)}/{len(target)}"
    )
    return best_state


class DeviceFineTuneModule(pl.LightningModule):
    def __init__(self, config, distilled_state: Dict[str, torch.Tensor]):
        super().__init__()
        self.config = config

        self.mel = nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate,
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max,
            ),
        )
        self.mel_augment = AdaptiveSpecAugment(
            freq_mask_param=config.freqm,
            time_mask_param=config.timem,
            max_num_masks=config.specaug_max_masks,
            schedule=config.specaug_schedule,
            total_epochs=config.n_epochs,
        )

        base_model = build_student(config)
        base_model.load_state_dict(distilled_state, strict=True)

        # base_model is the distilled general model and remains unchanged. It is
        # used for unseen devices s4/s5/s6. The six known-device models are exact
        # copies and are then fine-tuned independently.
        self.multi_device_model = MultiDeviceModelContainer(base_model, TRAIN_DEVICES)
        self.active_device = None
        self.validation_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_outputs: List[Dict[str, torch.Tensor]] = []

    def set_active_device(self, device_name: str):
        if device_name not in TRAIN_DEVICES:
            raise ValueError(f"Unsupported train device: {device_name}")
        self.active_device = device_name

        # Train only the currently selected device copy. Keep the distilled base
        # and all other device copies fixed.
        for parameter in self.multi_device_model.parameters():
            parameter.requires_grad = False
        for parameter in self.multi_device_model.get_model_for_device(device_name).parameters():
            parameter.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Device {device_name}] trainable parameters: {n_trainable:,}")

    def on_train_epoch_start(self):
        self.mel_augment.set_epoch(self.current_epoch)

    def mel_forward(self, waveform: torch.Tensor, augment: bool) -> torch.Tensor:
        x = self.mel(waveform)
        if augment:
            x = self.mel_augment(x)
        return (x + 1e-5).log()

    def forward(self, waveform: torch.Tensor, devices: Sequence[str]):
        features = self.mel_forward(waveform, augment=False)
        return self.multi_device_model(features, devices)

    def configure_optimizers(self):
        if self.active_device is None:
            raise RuntimeError("Call set_active_device() before trainer.fit().")
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        waveform, _, labels, devices, _ = batch
        unique_devices = set(str(d) for d in devices)
        if unique_devices != {self.active_device}:
            raise RuntimeError(
                f"Device-specific loader for {self.active_device} produced devices {unique_devices}."
            )

        features = self.mel_forward(waveform, augment=True)
        if self.config.probabilistic_p > 0:
            features = probabilistic(
                features,
                self.config.probabilistic_mixstyle_prob,
                self.config.probabilistic_p,
                self.config.probabilistic_alpha,
                self.config.probabilistic_lambda_val,
                self.config.probabilistic_phi_val,
            )
        logits = self.multi_device_model.get_model_for_device(self.active_device)(features)
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.config.label_smoothing,
        )
        accuracy = logits.argmax(1).eq(labels).float().mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveform, _, labels, devices, _ = batch
        logits = self.forward(waveform, devices)
        sample_loss = F.cross_entropy(logits, labels, reduction="none")
        correct = logits.argmax(1).eq(labels)

        result: Dict[str, torch.Tensor] = {
            "loss_sum": sample_loss.sum(),
            "count": torch.as_tensor(labels.numel(), device=self.device),
            "correct": correct.sum(),
        }
        for label_index in range(len(LABEL_NAMES)):
            mask = labels.eq(label_index)
            result[f"label_count.{label_index}"] = mask.sum()
            result[f"label_correct.{label_index}"] = (correct & mask).sum()
        self.validation_outputs.append({k: v.detach().cpu() for k, v in result.items()})

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        stacked = {
            key: torch.stack([item[key] for item in self.validation_outputs])
            for key in self.validation_outputs[0]
        }
        count = stacked["count"].sum().clamp_min(1)
        logs = {
            "val/loss": stacked["loss_sum"].sum() / count,
            "val/acc": stacked["correct"].sum() / count,
        }
        class_accs = []
        for label_index, label_name in enumerate(LABEL_NAMES):
            label_count = stacked[f"label_count.{label_index}"].sum()
            label_correct = stacked[f"label_correct.{label_index}"].sum()
            if label_count > 0:
                class_acc = label_correct / label_count
                class_accs.append(class_acc)
                logs[f"val/acc.{label_name}"] = class_acc
        logs["val/macro_avg_acc"] = (
            torch.stack(class_accs).mean() if class_accs else torch.tensor(0.0)
        )
        self.log_dict(logs, prog_bar=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        waveform, _, labels, devices, _ = batch
        logits = self.forward(waveform, devices)
        sample_loss = F.cross_entropy(logits, labels, reduction="none")
        correct = logits.argmax(1).eq(labels)

        result: Dict[str, torch.Tensor] = {
            "loss_sum": sample_loss.sum(),
            "count": torch.as_tensor(labels.numel(), device=self.device),
            "correct": correct.sum(),
        }
        for device_name in ALL_DEVICES:
            mask = torch.as_tensor(
                [str(d) == device_name for d in devices],
                device=self.device,
                dtype=torch.bool,
            )
            result[f"device_count.{device_name}"] = mask.sum()
            result[f"device_correct.{device_name}"] = (correct & mask).sum()
            result[f"device_loss.{device_name}"] = sample_loss[mask].sum()
        for label_index in range(len(LABEL_NAMES)):
            mask = labels.eq(label_index)
            result[f"label_count.{label_index}"] = mask.sum()
            result[f"label_correct.{label_index}"] = (correct & mask).sum()
        self.test_outputs.append({k: v.detach().cpu() for k, v in result.items()})

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        stacked = {
            key: torch.stack([item[key] for item in self.test_outputs])
            for key in self.test_outputs[0]
        }
        total_count = stacked["count"].sum().clamp_min(1)
        logs: Dict[str, torch.Tensor] = {
            "test/loss": stacked["loss_sum"].sum() / total_count,
            "test/acc": stacked["correct"].sum() / total_count,
        }

        group_correct = {group: torch.tensor(0.0) for group in set(DEVICE_GROUPS.values())}
        group_count = {group: torch.tensor(0.0) for group in set(DEVICE_GROUPS.values())}
        for device_name in ALL_DEVICES:
            count = stacked[f"device_count.{device_name}"].sum()
            correct = stacked[f"device_correct.{device_name}"].sum()
            loss_sum = stacked[f"device_loss.{device_name}"].sum()
            safe_count = count.clamp_min(1)
            logs[f"test/acc.{device_name}"] = correct / safe_count
            logs[f"test/loss.{device_name}"] = loss_sum / safe_count
            logs[f"test/cnt.{device_name}"] = count
            group = DEVICE_GROUPS[device_name]
            group_correct[group] += correct
            group_count[group] += count

        for group in group_count:
            logs[f"test/acc.{group}"] = group_correct[group] / group_count[group].clamp_min(1)

        class_accs = []
        for label_index, label_name in enumerate(LABEL_NAMES):
            count = stacked[f"label_count.{label_index}"].sum()
            correct = stacked[f"label_correct.{label_index}"].sum()
            class_acc = correct / count.clamp_min(1)
            logs[f"test/acc.{label_name}"] = class_acc
            class_accs.append(class_acc)
        logs["test/macro_avg_acc"] = torch.stack(class_accs).mean()
        self.log_dict(logs)
        print(
            f"[Final] acc={float(logs['test/acc']):.4f}; "
            f"macro_avg_acc={float(logs['test/macro_avg_acc']):.4f}"
        )
        self.test_outputs.clear()


def restore_best_checkpoint(module: DeviceFineTuneModule, checkpoint_path: str):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to restore device best checkpoint. missing={missing}, unexpected={unexpected}"
        )
    print(f"[Restore] {checkpoint_path}")


def save_combined_checkpoint(module: DeviceFineTuneModule, config, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": vars(config),
            "distilled_general_ckpt": str(Path(config.base_ckpt).resolve()),
            "known_device_models": TRAIN_DEVICES,
            "unseen_device_fallback": "base_model",
        },
        output,
    )
    print(f"[Saved] combined multi-device checkpoint: {output}")


def train(config):
    if config.subset != 25:
        raise ValueError("DCASE 2025 Task 1 subset must be 25%.")

    os.environ.setdefault("WANDB_BASE_URL", "https://api.bandw.top")
    os.environ["WANDB_MODE"] = config.wandb_mode
    pl.seed_everything(config.seed, workers=True)

    probe_model = build_student(config)
    distilled_state = extract_student_state_dict(config.base_ckpt, probe_model)
    del probe_model

    module = DeviceFineTuneModule(config, distilled_state)
    logger = WandbLogger(
        project=config.project_name,
        name=config.experiment_name,
        notes="Device-specific fine-tuning initialized from the sys4-distilled tiny general model.",
        tags=["DCASE25", "after-KD", "device-specific", "tiny-student"],
        config=vars(config),
    )

    # Verify one-device complexity; the teacher is absent in this stage.
    probe_dataset = get_test_set(device="a")
    probe_loader = DataLoader(probe_dataset, batch_size=1, shuffle=False, num_workers=0)
    waveform = next(iter(probe_loader))[0]
    with torch.no_grad():
        feature_shape = module.mel_forward(waveform, augment=False).shape
    one_model = copy.deepcopy(module.multi_device_model.get_model_for_device("a"))
    macs, parameter_bytes = complexity.get_torch_macs_memory(one_model, input_size=feature_shape)
    logger.experiment.config.update({
        "One_Device_Model_MACs": int(macs),
        "One_Device_Model_Parameter_Bytes": int(parameter_bytes),
        "Initialization": "distilled general student",
    })
    print(f"[One model] MACs={macs:,}; parameter bytes={parameter_bytes:,}")

    roll_samples = int(config.orig_sample_rate * config.roll_sec)
    device_checkpoint_root = Path(config.output_dir) / "per_device"
    device_checkpoint_root.mkdir(parents=True, exist_ok=True)

    for device_name in TRAIN_DEVICES:
        print("\n" + "=" * 72)
        print(f"Fine-tuning distilled student for device {device_name.upper()}")
        print("=" * 72)

        module.set_active_device(device_name)
        train_dataset = get_training_set(config.subset, device=device_name, roll=roll_samples)
        validation_dataset = get_test_set(device=device_name)
        train_loader = DataLoader(
            train_dataset,
            worker_init_fn=worker_init_fn,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            worker_init_fn=worker_init_fn,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=device_checkpoint_root / device_name,
            monitor="val/macro_avg_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename=f"{device_name}" + "-{epoch:02d}-{val_macro_avg_acc:.4f}",
            auto_insert_metric_name=False,
        )
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val/macro_avg_acc",
            mode="max",
            patience=config.early_stop_patience,
            min_delta=config.early_stop_min_delta,
        )
        trainer = pl.Trainer(
            max_epochs=config.n_epochs,
            logger=logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=config.precision,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            callbacks=[checkpoint_callback, early_stopping],
            gradient_clip_val=config.gradient_clip_val,
            log_every_n_steps=config.log_every_n_steps,
        )
        trainer.fit(module, train_loader, validation_loader)
        restore_best_checkpoint(module, checkpoint_callback.best_model_path)

    # Keep the distilled base model untouched for s4/s5/s6 and save all six tuned
    # copies together in one deployment/training checkpoint.
    save_combined_checkpoint(module, config, config.output_ckpt)

    full_test_loader = DataLoader(
        get_test_set(device=None),
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    tester = pl.Trainer(
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config.precision,
    )
    tester.test(module, dataloaders=full_test_loader)
    wandb.finish()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune per-device models after sys4 knowledge distillation"
    )
    parser.add_argument(
        "--base_ckpt",
        type=str,
        required=True,
        help="Best/last checkpoint produced by train_base_kd_sys4.py",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/device_after_kd")
    parser.add_argument(
        "--output_ckpt",
        type=str,
        default="./outputs/device_after_kd/multi_device_after_kd.ckpt",
    )

    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="sys4_kd_then_device_ft")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--subset", type=int, default=25)

    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channels_multiplier", type=float, default=1.8)
    parser.add_argument("--expansion_rate", type=float, default=2.1)
    parser.add_argument("--fuse_type", choices=["AFF", "iAFF", "DAF"], default="AFF")
    parser.add_argument("--conv_type", type=str, default="shuffle_pw")
    parser.add_argument("--pw_groups", type=int, default=4)
    parser.add_argument("--group_conv_groups", type=int, default=4)
    parser.add_argument("--sparse_ratio", type=float, default=0.5)
    parser.add_argument("--use_asym_dw", type=str2bool, default=True)
    parser.add_argument("--dw_activation", choices=["silu", "relu"], default="silu")
    parser.add_argument("--use_maxavg_stage_fusion", type=str2bool, default=True)
    parser.add_argument("--asym_dw_freq", type=int, default=3)
    parser.add_argument("--asym_dw_time", type=int, default=3)
    parser.add_argument("--large_asym_dw_freq", type=int, default=5)
    parser.add_argument("--large_asym_dw_time", type=int, default=3)
    parser.add_argument("--large_kernel_blocks", type=str, default="b4,b5")

    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--roll_sec", type=float, default=0.1)

    parser.add_argument("--probabilistic_mixstyle_prob", type=float, default=0.7)
    parser.add_argument("--probabilistic_p", type=float, default=0.2)
    parser.add_argument("--probabilistic_alpha", type=float, default=0.3)
    parser.add_argument("--probabilistic_lambda_val", type=float, default=0.05)
    parser.add_argument("--probabilistic_phi_val", type=float, default=0.05)

    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)
    parser.add_argument("--freqm", type=int, default=12)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--specaug_max_masks", type=int, default=1)
    parser.add_argument("--specaug_schedule", choices=["linear", "step", "constant"], default="constant")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
