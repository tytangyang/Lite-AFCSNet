import copy
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pytorch_lightning as pl
import transformers
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# Local imports
from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers.utils import probabilistic
from helpers import complexity
from models.net import get_two_stage_model
from models.multi_device_model import MultiDeviceModelContainer

class AdaptiveSpecAugment(nn.Module):
    """自适应 SpecAugment，可以根据训练进度调整强度"""
    def __init__(self, freq_mask_param, time_mask_param, 
                 max_num_masks=2, schedule="linear", total_epochs=150):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.max_num_masks = max_num_masks
        self.schedule = schedule
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """设置当前 epoch，用于自适应增强"""
        self.current_epoch = epoch
    
    def forward(self, x):
        # 根据训练进度计算当前应应用的掩蔽次数
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
        else:  # constant
            num_masks = self.max_num_masks
        
        # 随机决定实际应用几次（不超过 num_masks）
        actual_num_freq = torch.randint(0, num_masks + 1, (1,)).item()
        actual_num_time = torch.randint(0, num_masks + 1, (1,)).item()
        
        # 应用频率掩蔽
        for _ in range(actual_num_freq):
            x = T.FrequencyMasking(self.freq_mask_param, iid_masks=True)(x)
        
        # 应用时域掩蔽
        for _ in range(actual_num_time):
            x = T.TimeMasking(self.time_mask_param, iid_masks=True)(x)
        
        return x

class PLModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training the DCASE'25 baseline model (fine-tune general model for specific devices).
    """
    def __init__(self, config, base_model_state_dict=None):
        super().__init__()
        self.config = config

        # ----- Preprocessing Pipeline -----
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
            )
        )
        
        # 使用改进的 AdaptiveSpecAugment
        self.mel_augment = AdaptiveSpecAugment(
            freq_mask_param=config.freqm,
            time_mask_param=config.timem,
            max_num_masks=config.specaug_max_masks,  # 从2开始，逐渐增加
            schedule=config.specaug_schedule,  # "linear", "step", "constant"
            total_epochs=config.n_epochs
        )

        # ----- Device Definitions -----
        self.train_device_ids = ['a', 'b', 'c', 's1', 's2', 's3']  # devices used for fine-tuning
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]
        # Group devices into real/seen/unseen
        self.device_groups = {
            'a': "real", 'b': "real", 'c': "real",
            's1': "seen", 's2': "seen", 's3': "seen",
            's4': "unseen", 's5': "unseen", 's6': "unseen"
        }

        # ----- Base Model + Multi-Device Container -----
        base_model = get_two_stage_model(
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channels_multiplier=config.channels_multiplier,
            expansion_rate=config.expansion_rate
        )

        # Load the pre-trained model weights if provided
        if base_model_state_dict is not None:
            base_model.load_state_dict(base_model_state_dict, strict=True)

        # Wrap the base model in a MultiDeviceModelContainer for device-specific fine-tuning
        self.multi_device_model = MultiDeviceModelContainer(
            base_model,
            self.train_device_ids
        )

        # Lightning containers for step outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Tracks the current device for validation logging
        self.validation_device = None

    def on_train_epoch_start(self):
        """在每个训练 epoch 开始时更新增强参数"""
        super().on_train_epoch_start()
        if hasattr(self.mel_augment, 'set_epoch'):
            self.mel_augment.set_epoch(self.current_epoch)

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x, devices):
        """
        :param x: batch of raw audio signals (waveforms)
        :param devices: devices corresponding to waveforms
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.multi_device_model(x, devices)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, _, labels, devices, _ = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config.probabilistic_p > 0:
            # frequency mixstyle
            x = probabilistic(x, self.config.probabilistic_mixstyle_prob, self.config.probabilistic_p, self.config.probabilistic_alpha, self.config.probabilistic_lambda_val, self.config.probabilistic_phi_val)
        y_hat = self.multi_device_model(x, devices)
        loss = F.cross_entropy(y_hat, labels)

        # Log learning rate
        self.log(f"lr.{devices[0]}", self.trainer.optimizers[0].param_groups[0]["lr"])

        # Log training loss
        self.log(f"train/loss.{devices[0]}", loss.detach().cpu())

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        assert len(set(devices)) == 1, "Each validation batch should contain samples from a single device."

        y_hat = self.forward(x, devices)
        samples_loss = F.cross_entropy(y_hat, labels)

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "n_correct": n_correct,
            "n_pred": torch.tensor(len(labels), device=self.device),
            "devloss": samples_loss.sum(),
            "devn_correct": n_correct,
            "devcnt": torch.tensor(len(devices), device=self.device)
        }

        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

        # Keep track of the device used in this validation round
        self.validation_device = devices[0]

    def on_validation_epoch_end(self):
        # Flatten step outputs
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack results
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        logs = {}

        # Log device stats
        dev_loss = outputs["devloss"].sum()
        dev_cnt = outputs["devcnt"].sum()
        dev_correct = outputs["devn_correct"].sum()
        device_name = self.validation_device
        logs[f"loss.{device_name}"] = dev_loss / dev_cnt
        logs[f"acc.{device_name}"] = dev_correct / dev_cnt
        logs[f"cnt.{device_name}"] = dev_cnt.float()

        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        self.validation_step_outputs.clear()
        self.validation_device = None

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch

        # Switch model to FP16 to meet memory constraints
        self.multi_device_model.half()
        x = self.mel_forward(x)
        x = x.half()

        y_hat = self.multi_device_model(x, devices)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.tensor(len(labels), device=self.device)
        }

        # Per-device stats
        for dev_id in self.device_ids:
            results[f"devloss.{dev_id}"] = torch.tensor(0., device=self.device)
            results[f"devcnt.{dev_id}"] = torch.tensor(0., device=self.device)
            results[f"devn_correct.{dev_id}"] = torch.tensor(0., device=self.device)

        for i, dev_name in enumerate(dev_names):
            results[f"devloss.{dev_name}"] += samples_loss[i]
            results[f"devn_correct.{dev_name}"] += n_correct_per_sample[i]
            results[f"devcnt.{dev_name}"] += 1

        # Per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.tensor(0., device=self.device)

        for i, lbl_idx in enumerate(labels):
            lbl_name = self.label_ids[lbl_idx]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})

    def on_test_epoch_end(self):
        # Flatten outputs
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack results
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        # Per-device logs
        for dev_id in self.device_ids:
            dev_loss = outputs[f"devloss.{dev_id}"].sum()
            dev_cnt = outputs[f"devcnt.{dev_id}"].sum()
            dev_correct = outputs[f"devn_correct.{dev_id}"].sum()
            logs[f"loss.{dev_id}"] = dev_loss / dev_cnt
            logs[f"acc.{dev_id}"] = dev_correct / dev_cnt
            logs[f"cnt.{dev_id}"] = dev_cnt

            # Device groups
            grp = self.device_groups[dev_id]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + dev_correct
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt
            logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}", 0.) + dev_loss

        # Aggregate device group stats
        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        # Label-level stats
        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
            logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt

        # Macro-average accuracy across all labels
        logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        """
        Predict step for evaluation set inference.

        Args:
            eval_batch: (waveforms, filenames).
            batch_idx: Batch index.
            dataloader_idx: Index of the dataloader in a multi-dataloader scenario.

        Returns:
            (list[str], torch.Tensor): Filenames and logits predictions.
        """
        x, files = eval_batch

        self.multi_device_model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.multi_device_model(x)
        return files, y_hat


def train(config):
    """
    Fine-tune a pre-trained baseline model on specific devices, then test on all devices.

    Args:
        config (argparse.Namespace): Contains hyperparameters and paths.
    """
    # Load pre-trained checkpoint if provided
    if config.ckpt_id is not None:
        ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
        assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
        ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
        assert os.path.exists(ckpt_file), (
            f"No such file: {ckpt_file}. Adjust 'ckpt_id'."
        )
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # Extract only the base model weights (prefixed by 'model.')
        base_model_state_dict = {
            k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")
        }
    else:
        base_model_state_dict = None

    # Create the Lightning module with multi-device model container
    pl_module = PLModule(config, base_model_state_dict=base_model_state_dict)

    # Validate subset selection
    assert config.subset == 25, (
        "In DCASE`25 subset is fixed to 25%."
    )
    roll_samples = config.orig_sample_rate * config.roll_sec

    print("Hint: In Weights & Biases, switch X-axis to `trainer/global_step` for aligned plots.")

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'25 Task 1.",
        tags=["DCASE25"],
        config=config,
        name=config.experiment_name
    )

    for device_id in pl_module.train_device_ids:
        print("\n" + "=" * 60)
        print(f"?? Fine-tuning for device: **{device_id.upper()}**".center(60))
        print("=" * 60 + "\n")

        # load device-specific datasets
        train_ds = get_training_set(config.subset, device=device_id, roll=roll_samples)
        train_dl = DataLoader(
            dataset=train_ds,
            worker_init_fn=worker_init_fn,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=True
        )
        test_ds = get_test_set(device=device_id)
        test_dl = DataLoader(
            dataset=test_ds,
            worker_init_fn=worker_init_fn,
            num_workers=config.num_workers,
            batch_size=config.batch_size
        )

        # Log model complexity for this device's model
        sample = next(iter(test_dl))[0][0].unsqueeze(0)
        shape = pl_module.mel_forward(sample).size()
        model = pl_module.multi_device_model.get_model_for_device(device_id)
        half_model = copy.deepcopy(model)
        half_model.half()  # we plan to use the model with 16-bit parameter precision
        macs, params_bytes = complexity.get_torch_macs_memory(half_model, input_size=shape)
        wandb_logger.experiment.config[f"MACs_{device_id}_model"] = macs
        wandb_logger.experiment.config[f"Parameters_Bytes_{device_id}_model"] = params_bytes

        # Trainer for device-specific fine-tuning
        trainer = pl.Trainer(
            max_epochs=config.n_epochs,
            logger=wandb_logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=config.precision,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)]
        )

        trainer.fit(pl_module, train_dl, test_dl)

    # After fine-tuning, test on *all* devices
    test_dl = DataLoader(
        dataset=get_test_set(device=None),
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        precision=config.precision
    )
    trainer.test(pl_module, dataloaders=test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DCASE 25 argument parser")

    # General arguments
    parser.add_argument("--ckpt_id", type=str, default="/root/autodl-tmp/dcase2025_task1_baseline-main/DCASE25_Task1/dq8fwh5a", help="Checkpoint ID for loading base model")
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="17.2KB_device")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)

    # Evaluate vs. train
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation/prediction on the eval set")

    # in DCASE`25 Task 1 only the 25% subset can be used
    parser.add_argument("--subset", type=int, default=25)

    # Model hyperparameters
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channels_multiplier", type=float, default=1.8)
    parser.add_argument("--expansion_rate", type=float, default=2.1)

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--probabilistic_mixstyle_prob", type=float, default=0.7)
    parser.add_argument("--probabilistic_p", type=float, default=0.4)
    parser.add_argument("--probabilistic_alpha", type=float, default=0.3)
    parser.add_argument("--probabilistic_lambda_val", type=float, default=0.05)
    parser.add_argument("--probabilistic_phi_val", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--roll_sec", type=float, default=0.1)

    # Learning rate schedule
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--warmup_steps", type=int, default=200)

    # Spectrogram parameters
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--freqm", type=int, default=24)  # 从48减小到24，更温和
    parser.add_argument("--timem", type=int, default=0)   # 暂时不使用时域掩蔽
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)
    
    # SpecAugment 相关参数
    parser.add_argument("--specaug_max_masks", type=int, default=2, 
                       help="最大掩蔽次数（频率+时间）")
    parser.add_argument("--specaug_schedule", type=str, default="linear",
                       choices=["linear", "step", "constant"],
                       help="增强强度调度策略")

    args = parser.parse_args()

    if args.evaluate:
        raise NotImplementedError("Evaluation routine will be released together"
                                  " with the evaluation set on 1st of June!")
    else:
        train(args)
