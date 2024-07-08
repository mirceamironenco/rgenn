import argparse
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)


class NativeScaler:
    # Adapted from timm/utils/cuda.py
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def seed_everything(seed: int = 100):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMetric:
    count: int = 0
    last_value: float = 0.0
    total: float = 0.0
    average: float = 0.0

    def reset(self) -> None:
        self.count = 0
        self.last_value = 0.0
        self.total = 0.0
        self.average = 0.0

    def update(self, value: float, count: int = 1) -> None:
        self.count += count
        self.last_value = value
        self.total += value * count
        self.average = self.total / self.count


def reduce_tensor(tensor: torch.Tensor, n: int) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed(config: argparse.Namespace) -> torch.device:
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0

    device = getattr(config, "device", "cuda")
    device_type, *device_idx = device.split(":", maxsplit=1)
    if device_type == "cuda":
        assert (
            torch.cuda.is_available()
        ), f"CUDA is not available but {device_type} was specified."

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        # DDP by torchrun only
        dist_url = "env://"
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method=dist_url)
        assert dist.is_initialized()
        world_size = dist.get_world_size()
        local_rank = get_local_rank()
        global_rank = dist.get_rank()

        if device_type != "cpu":
            if device_idx:
                _logger.warning(
                    f"device index {device_idx[0]} removed from specified ({device})."
                )
            device = f"{device_type}:{local_rank}"

    if device.startswith("cuda:"):
        torch.cuda.set_device(device)

    config.distributed = distributed
    config.world_size = world_size
    config.rank = global_rank
    config.local_rank = local_rank
    config.device = device

    if config.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {config.rank}, total {config.world_size}, device {config.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({config.device}).")
    assert config.rank >= 0

    return torch.device(config.device)


def is_primary(config: argparse.Namespace) -> bool:
    return config.rank == 0


def configure_backends(config) -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = getattr(config, "allow_tf32", True)
        torch.backends.cudnn.benchmark = True
