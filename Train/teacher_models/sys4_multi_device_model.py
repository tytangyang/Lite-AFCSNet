import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn


class MultiDeviceModelContainer(nn.Module):
    def __init__(self, base_model: nn.Module, devices: Sequence[str]):
        super().__init__()
        self.base_model = base_model
        self.devices = list(devices)
        self.device_models = nn.ModuleDict(
            {device: copy.deepcopy(base_model) for device in self.devices}
        )

    def get_model_for_device(self, device_name: str) -> nn.Module:
        if device_name in self.device_models:
            return self.device_models[device_name]
        return self.base_model

    def forward(self, x: torch.Tensor, devices: Optional[Sequence[str]] = None):
        if devices is None:
            return self.base_model(x)
        if len(devices) != x.size(0):
            raise ValueError(
                f"devices length {len(devices)} does not match batch size {x.size(0)}"
            )

        # Grouped routing is much faster than one forward per sample.
        device_names = [str(d) for d in devices]
        outputs = None
        for device_name in dict.fromkeys(device_names):
            indices = [i for i, d in enumerate(device_names) if d == device_name]
            index = torch.as_tensor(indices, device=x.device, dtype=torch.long)
            model = self.get_model_for_device(device_name)
            part = model(x.index_select(0, index))
            if outputs is None:
                outputs = torch.empty(
                    (x.size(0), part.size(1)),
                    device=part.device,
                    dtype=part.dtype,
                )
            outputs.index_copy_(0, index, part)
        return outputs
