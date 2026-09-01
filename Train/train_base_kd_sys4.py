import argparse
import os
from typing import Dict, List

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
from teacher_models.sys4_teacher import Sys4Teacher


DEVICE_IDS = ["a", "b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]
LABEL_IDS = [
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
        max_num_masks: int = 2,
        schedule: str = "linear",
        total_epochs: int = 150,
    ) -> None:
        super().__init__()
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.max_num_masks = int(max_num_masks)
        self.schedule = schedule
        self.total_epochs = int(total_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.schedule == "linear":
            progress = min(1.0, self.current_epoch / max(1, self.total_epochs - 20))
            num_masks = max(1, int(self.max_num_masks * progress))
        elif self.schedule == "step":
            if self.current_epoch < self.total_epochs // 3:
                num_masks = 1
            elif self.current_epoch < 2 * self.total_epochs // 3:
                num_masks = 2
            else:
                num_masks = self.max_num_masks
        else:
            num_masks = self.max_num_masks

        n_freq = int(torch.randint(0, num_masks + 1, (1,)).item())
        n_time = int(torch.randint(0, num_masks + 1, (1,)).item())
        for _ in range(n_freq):
            x = T.FrequencyMasking(self.freq_mask_param, iid_masks=True)(x)
        for _ in range(n_time):
            x = T.TimeMasking(self.time_mask_param, iid_masks=True)(x)
        return x


class PLModule(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self._validate_teacher_frontend()

        # This frontend exactly matches both the current student training code and
        # the first-place sys4 training/inference configuration.
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

        # Current 17.2KB / 9.67MMAC student. Defaults reproduce the uploaded net.py.
        self.model = get_two_stage_model(
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
                block.strip() for block in config.large_kernel_blocks.split(",") if block.strip()
            ),
        )

        teacher = Sys4Teacher(
            checkpoint_path=config.teacher_ckpt,
            use_device_models=config.teacher_use_device_models,
            n_classes=10,
            in_channels=1,
            base_channels=32,
            channels_multiplier=1.8,
            expansion_rate=2.1,
        )

        # Deliberately bypass nn.Module registration. Therefore sys4 parameters are
        # never saved in the student's Lightning checkpoint and do not change the
        # deployable model size.
        object.__setattr__(self, "_teacher", teacher)

        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_step_outputs: List[Dict[str, torch.Tensor]] = []

    def _validate_teacher_frontend(self) -> None:
        expected = {
            "sample_rate": 32000,
            "window_length": 3072,
            "hop_length": 500,
            "n_fft": 4096,
            "n_mels": 256,
            "f_min": 0,
            "f_max": None,
        }
        actual = {key: getattr(self.config, key) for key in expected}
        mismatches = {
            key: (actual[key], value)
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise ValueError(
                "The first-place sys4 checkpoint requires its official log-mel frontend. "
                f"Mismatched values: {mismatches}"
            )

    def on_fit_start(self) -> None:
        self._teacher.to(self.device, dtype=torch.float32)
        self._teacher.eval()

    def on_train_epoch_start(self) -> None:
        self.mel_augment.set_epoch(self.current_epoch)
        self._teacher.eval()

    def waveform_to_mel_power(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel(waveform)

    @staticmethod
    def mel_power_to_log(mel_power: torch.Tensor) -> torch.Tensor:
        return (mel_power + 1e-5).log()

    def student_features(self, waveform: torch.Tensor, augment: bool) -> torch.Tensor:
        mel_power = self.waveform_to_mel_power(waveform)
        # Preserve the uploaded train_base.py ordering: SpecAugment is applied to
        # linear mel power, then log compression, then Fre-MixStyle/RandIN.
        if augment:
            mel_power = self.mel_augment(mel_power)
        return self.mel_power_to_log(mel_power)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(self.student_features(waveform, augment=False))

    @staticmethod
    def distillation_kl(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (temperature ** 2)

    def effective_kd_alpha(self) -> float:
        if self.config.kd_warmup_epochs <= 0:
            return float(self.config.kd_alpha)
        progress = min(
            1.0,
            float(self.current_epoch + 1) / float(self.config.kd_warmup_epochs),
        )
        return float(self.config.kd_alpha) * progress

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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

        # One mel computation is shared. Teacher receives the clean official log-mel;
        # student receives the same sample with the original training augmentations.
        mel_power = self.waveform_to_mel_power(waveform)
        teacher_log_mel = self.mel_power_to_log(mel_power)

        student_mel_power = self.mel_augment(mel_power.clone())
        student_log_mel = self.mel_power_to_log(student_mel_power)
        if self.config.probabilistic_p > 0:
            student_log_mel = probabilistic(
                student_log_mel,
                self.config.probabilistic_mixstyle_prob,
                self.config.probabilistic_p,
                self.config.probabilistic_alpha,
                self.config.probabilistic_lambda_val,
                self.config.probabilistic_phi_val,
            )

        student_logits = self.model(student_log_mel)
        with torch.inference_mode():
            teacher_logits = self._teacher(teacher_log_mel.float(), devices)

        ce_loss = F.cross_entropy(student_logits, labels)
        kd_loss = self.distillation_kl(
            student_logits.float(),
            teacher_logits.float(),
            self.config.kd_temperature,
        )
        alpha = self.effective_kd_alpha()
        total_loss = (1.0 - alpha) * ce_loss + alpha * kd_loss

        teacher_acc = teacher_logits.argmax(1).eq(labels).float().mean()
        student_acc = student_logits.argmax(1).eq(labels).float().mean()
        agreement = teacher_logits.argmax(1).eq(student_logits.argmax(1)).float().mean()

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/ce_loss", ce_loss)
        self.log("train/kd_loss", kd_loss)
        self.log("train/kd_alpha", alpha)
        self.log("train/teacher_acc", teacher_acc)
        self.log("train/student_acc", student_acc)
        self.log("train/student_teacher_agreement", agreement)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("epoch", float(self.current_epoch))
        return total_loss

    def _shared_eval_step(self, batch) -> Dict[str, torch.Tensor]:
        waveform, _, labels, devices, _ = batch
        logits = self.forward(waveform)
        sample_losses = F.cross_entropy(logits, labels, reduction="none")
        predictions = logits.argmax(dim=1)
        correct = predictions.eq(labels)

        results: Dict[str, torch.Tensor] = {
            "loss": sample_losses.mean(),
            "n_correct": correct.sum(),
            "n_pred": torch.as_tensor(labels.numel(), device=self.device),
        }
        for device_name in DEVICE_IDS:
            results[f"devloss.{device_name}"] = torch.zeros((), device=self.device)
            results[f"devcnt.{device_name}"] = torch.zeros((), device=self.device)
            results[f"devn_correct.{device_name}"] = torch.zeros((), device=self.device)
        for index, device_name in enumerate(devices):
            device_name = str(device_name)
            results[f"devloss.{device_name}"] += sample_losses[index]
            results[f"devcnt.{device_name}"] += 1
            results[f"devn_correct.{device_name}"] += correct[index]

        for label_name in LABEL_IDS:
            results[f"lblloss.{label_name}"] = torch.zeros((), device=self.device)
            results[f"lblcnt.{label_name}"] = torch.zeros((), device=self.device)
            results[f"lbln_correct.{label_name}"] = torch.zeros((), device=self.device)
        for index, label_index in enumerate(labels):
            label_name = LABEL_IDS[int(label_index)]
            results[f"lblloss.{label_name}"] += sample_losses[index]
            results[f"lblcnt.{label_name}"] += 1
            results[f"lbln_correct.{label_name}"] += correct[index]

        return {key: value.detach().cpu() for key, value in results.items()}

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self._shared_eval_step(batch))

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self._shared_eval_step(batch))

    def _aggregate_epoch(self, outputs, prefix: str) -> None:
        if not outputs:
            return
        stacked = {
            key: torch.stack([item[key] for item in outputs])
            for key in outputs[0]
        }
        logs: Dict[str, torch.Tensor] = {
            "loss": stacked["loss"].mean(),
            "acc": stacked["n_correct"].sum() / stacked["n_pred"].sum().clamp_min(1),
        }

        for device_name in DEVICE_IDS:
            count = stacked[f"devcnt.{device_name}"].sum()
            safe_count = count.clamp_min(1)
            loss_sum = stacked[f"devloss.{device_name}"].sum()
            correct_sum = stacked[f"devn_correct.{device_name}"].sum()
            logs[f"loss.{device_name}"] = loss_sum / safe_count
            logs[f"acc.{device_name}"] = correct_sum / safe_count
            logs[f"cnt.{device_name}"] = count

            group = DEVICE_GROUPS[device_name]
            logs[f"_correct.{group}"] = logs.get(f"_correct.{group}", torch.tensor(0.0)) + correct_sum
            logs[f"_count.{group}"] = logs.get(f"_count.{group}", torch.tensor(0.0)) + count
            logs[f"_loss_sum.{group}"] = logs.get(f"_loss_sum.{group}", torch.tensor(0.0)) + loss_sum

        for group in set(DEVICE_GROUPS.values()):
            group_count = logs[f"_count.{group}"].clamp_min(1)
            logs[f"acc.{group}"] = logs[f"_correct.{group}"] / group_count
            logs[f"loss.{group}"] = logs[f"_loss_sum.{group}"] / group_count
            del logs[f"_correct.{group}"]
            del logs[f"_count.{group}"]
            del logs[f"_loss_sum.{group}"]

        label_accs = []
        for label_name in LABEL_IDS:
            count = stacked[f"lblcnt.{label_name}"].sum()
            safe_count = count.clamp_min(1)
            loss_sum = stacked[f"lblloss.{label_name}"].sum()
            correct_sum = stacked[f"lbln_correct.{label_name}"].sum()
            logs[f"loss.{label_name}"] = loss_sum / safe_count
            logs[f"acc.{label_name}"] = correct_sum / safe_count
            logs[f"cnt.{label_name}"] = count
            label_accs.append(logs[f"acc.{label_name}"])
        logs["macro_avg_acc"] = torch.stack(label_accs).mean()

        self.log_dict({f"{prefix}/{key}": value for key, value in logs.items()})
        outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self._aggregate_epoch(self.validation_step_outputs, "val")

    def on_test_epoch_end(self) -> None:
        self._aggregate_epoch(self.test_step_outputs, "test")


def train(config) -> None:
    os.environ.setdefault("WANDB_BASE_URL", "https://api.bandw.top")
    os.environ["WANDB_MODE"] = config.wandb_mode
    pl.seed_everything(config.seed, workers=True)

    logger = WandbLogger(
        project=config.project_name,
        name=config.experiment_name,
        notes="17.2KB student distilled directly from first-place sys4.ckpt.",
        tags=["DCASE25", "sys4", "device-aware-KD", "tiny-student"],
        config=vars(config),
    )

    if config.subset != 25:
        raise ValueError("DCASE 2025 Task 1 training subset must be 25%.")

    roll_samples = int(config.orig_sample_rate * config.roll_sec)
    train_dataset = get_training_set(config.subset, device=None, roll=roll_samples)
    validation_dataset = get_test_set(device=None)

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

    module = PLModule(config)

    sample_waveform = next(iter(validation_loader))[0][0].unsqueeze(0)
    with torch.no_grad():
        sample_features = module.student_features(sample_waveform, augment=False)
    macs, parameter_bytes = complexity.get_torch_macs_memory(
        module.model,
        input_size=sample_features.size(),
    )
    logger.experiment.config.update({
        "Student_MACs": int(macs),
        "Student_Parameters_Bytes": int(parameter_bytes),
        "Teacher_saved_in_student_ckpt": False,
    })
    print(f"[Student] MACs={macs:,}; parameter bytes={parameter_bytes:,}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/macro_avg_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="student-{epoch:03d}-{val_macro_avg_acc:.4f}",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config.precision,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
    )
    trainer.fit(module, train_loader, validation_loader)
    trainer.test(ckpt_path="best", dataloaders=validation_loader)
    wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct KD: first-place sys4.ckpt -> current tiny DCASE student"
    )
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--teacher_use_device_models", type=str2bool, default=True)

    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="sys4_direct_kd_a010_t3")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=20)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--subset", type=int, default=25)

    # Student model: defaults match the uploaded 17.2KB/9.67MMAC net.py path.
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channels_multiplier", type=float, default=1.8)
    parser.add_argument("--expansion_rate", type=float, default=2.1)
    parser.add_argument("--fuse_type", choices=["AFF", "iAFF", "DAF"], default="AFF")
    parser.add_argument("--conv_type", default="shuffle_pw")
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

    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--roll_sec", type=float, default=0.1)

    parser.add_argument("--probabilistic_mixstyle_prob", type=float, default=0.7)
    parser.add_argument("--probabilistic_p", type=float, default=0.4)
    parser.add_argument("--probabilistic_alpha", type=float, default=0.3)
    parser.add_argument("--probabilistic_lambda_val", type=float, default=0.05)
    parser.add_argument("--probabilistic_phi_val", type=float, default=0.05)

    # Official sys4/student frontend.
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)
    parser.add_argument("--freqm", type=int, default=24)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--specaug_max_masks", type=int, default=2)
    parser.add_argument("--specaug_schedule", choices=["linear", "step", "constant"], default="linear")

    # Direct logit distillation.
    parser.add_argument("--kd_alpha", type=float, default=0.10,
                        help="Final KD weight: total=(1-alpha)*CE + alpha*KD")
    parser.add_argument("--kd_temperature", type=float, default=3.0)
    parser.add_argument("--kd_warmup_epochs", type=int, default=30)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
