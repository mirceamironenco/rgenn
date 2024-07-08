import argparse
import functools
import logging
import time
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from rgenn.data import AffnistTransformed, AffnistUntransformed
from rgenn.groups import (
    GL2,
    SL2,
    SO2,
    SPD2,
    EquiDistantSO2,
    HaarUniformSOn,
    LogNormalSPD2,
    LogUniformSO2,
    MatrixManifold,
)
from rgenn.inr import SIREN, INRPartConfig, SIRENConfig
from rgenn.models import SamplingWrapper, GResNet
from rgenn.utils import (
    AverageMetric,
    NativeScaler,
    configure_backends,
    init_distributed,
    is_primary,
    reduce_tensor,
    seed_everything,
)

_logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("training")

    # Model
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--blocks_dim", type=int, default=40)
    parser.add_argument("--stem_kernel", type=int, default=5)
    parser.add_argument("--conv_kernel", type=int, default=5)
    parser.add_argument(
        "--global_pool", type=str, choices=["max", "mean"], default="max"
    )
    parser.add_argument("--act", type=str, choices=["relu", "gelu"], default="gelu")
    parser.add_argument(
        "--conv_norm",
        type=str,
        choices=["layernorm3d", "batchnorm3d", "layernorm3dg"],
        default="layernorm3d",
    )
    parser.add_argument(
        "--head_norm", type=str, choices=["layernorm", "batchnorm"], default="layernorm"
    )

    # INR
    parser.add_argument("--inr", type=str, choices=["siren", "wire"], default="siren")
    parser.add_argument("--inr_dim", type=int, default=60)
    parser.add_argument("--inr_nlayers", type=int, default=2)
    parser.add_argument("--siren_w0", type=float, default=10.0)
    parser.add_argument("--siren_first_w0", type=float, default=10.0)

    # Group
    parser.add_argument("--gsamples", type=int, default=10)
    parser.add_argument("--liegroup", type=str, choices=["sl2", "gl2"], default="sl2")
    parser.add_argument("--metric_alpha", type=float, default=1.0)

    parser.add_argument("--spd_bounds", type=float, default=0.1)
    parser.add_argument(
        "--ortho_sampler",
        type=str,
        choices=["equi_distant", "loguniform", "haar"],
        default="equi_distant",
    )
    parser.add_argument("--ortho_bounds", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
    )

    # Training & Misc
    parser.add_argument("--datadir", type=str, default="datasets")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_epoch_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torchcompile", type=str, default="inductor")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--amp_dtype", type=str, default="float16")
    parser.add_argument("--allow_tf32", action="store_true", default=False)

    args = parser.parse_args()
    return args


def create_group(config) -> MatrixManifold:
    # Configure SPD component
    spd2 = SPD2(traceless=config.liegroup == "sl2", metric_alpha=config.metric_alpha)
    spd_sampler = LogNormalSPD2(spd2, bounds=config.spd_bounds)

    # Configure orthogonal component
    so2 = SO2(metric_alpha=config.metric_alpha)
    sampler_cls = dict(
        equi_distant=EquiDistantSO2, loguniform=LogUniformSO2, haar=HaarUniformSOn
    )
    so2_sampler = sampler_cls[config.ortho_sampler](so2, bounds=config.ortho_bounds)

    # Instantiate larger group that uses product parametrization
    group_cls = dict(sl2=SL2, gl2=GL2)
    group = group_cls[config.liegroup](
        config.metric_alpha,
        spd2,
        so2,
        spd_sampler=spd_sampler,
        orthogonal_sampler=so2_sampler,
    )
    return group


def build_model(config) -> nn.Module:
    group = create_group(config)

    # Example configuration of INR which uses SIREN.
    siren_config = SIRENConfig(
        hidden_features=config.inr_dim,
        num_hidden=config.inr_nlayers,
        first_w0=config.siren_first_w0,
        w0=config.siren_w0,
        use_bias=True,
    )
    inr_config = INRPartConfig(
        kernel_class=SIREN,
        kernel_config=siren_config,
        hidden_features=siren_config.hidden_features,
        final_bias=True,
        final_gain=6.0,
        final_mixed_fans=False,
    )

    # Basic residual network
    model = GResNet(
        group=group,
        num_gsamples=config.gsamples,
        inr_cfg=inr_config,
        dims=[config.blocks_dim] * config.nlayers,
        in_chans=1,
        stem_kernel=config.stem_kernel,
        stem_padding=0,
        blocks_kernel=config.conv_kernel,
        num_classes=10,
        global_pool=config.global_pool,
        act_layer=config.act,
        norm_layer=config.head_norm,
        norm_layer_conv=config.conv_norm,
    )

    # Helper class used for sampling the group elements
    model = SamplingWrapper(
        model,
        num_gsamples=model.num_gsamples,
        group=group,
        sample_inference=True,
        deterministic=False,
    )
    return model


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    config,
    device=torch.device("cuda"),
    amp_autocast=None,
    loss_scaler=None,
):
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = AverageMetric()
    losses_m = AverageMetric()

    model.train()

    accum_steps = config.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in tqdm(
        enumerate(loader),
        total=len(loader),
        unit="batch",
        desc=f"Epoch {epoch}",
        disable=config.local_rank != 0,
        ncols=100,
    ):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        input, target = input.to(device), target.to(device)
        with model.no_sync() if (has_no_sync and not need_update) else nullcontext():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)

            if accum_steps > 1:
                loss /= accum_steps

            if loss_scaler is not None:
                loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=config.clip_grad,
                    parameters=model.parameters(),
                    create_graph=False,
                    need_update=need_update,
                )
            else:
                loss.backward()
                if need_update:
                    if config.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.clip_grad
                        )
                    optimizer.step()

        if not config.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            continue

        num_updates += 1
        optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % config.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if config.distributed:
                reduced_loss = reduce_tensor(loss.data, config.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                update_sample_count *= config.world_size

            if is_primary(config):
                tqdm.write(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  "
                    f"Loss: {losses_m.last_value:#.3g} ({losses_m.average:#.3g})  "
                    f"Time: {update_time_m.last_value:.3f}s, {update_sample_count / update_time_m.last_value:>7.2f}/s  "
                    f"({update_time_m.average:.3f}s, {update_sample_count / update_time_m.average:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                )
        update_sample_count = 0

    return dict([("loss", losses_m.average)])


def validate(
    epoch,
    model,
    loader,
    loss_fn,
    config,
    device=torch.device("cuda"),
    amp_autocast=nullcontext,
    log_suffix="",
):
    batch_time_m = AverageMetric()
    losses_m = AverageMetric()
    top1_m = AverageMetric()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in tqdm(
            enumerate(loader),
            total=len(loader),
            unit="batch",
            desc=f"Epoch {epoch}",
            disable=config.local_rank != 0,
            ncols=100,
        ):
            # for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input, target = input.to(device), target.to(device)

            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)

            acc1 = output.argmax(1).eq(target).float().mean()

            if config.distributed:
                reduced_loss = reduce_tensor(loss.data, config.world_size)
                acc1 = reduce_tensor(acc1, config.world_size)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if is_primary(config) and (
                last_batch or batch_idx % config.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                tqdm.write(
                    f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.last_value:.3f} ({batch_time_m.average:.3f})  "
                    f"Loss: {losses_m.last_value:>7.3f} ({losses_m.average:>6.3f})  "
                    f"Acc@1: {top1_m.last_value:>7.3f} ({top1_m.average:>7.3f})  "
                )

    metrics = dict([("loss", losses_m.average), ("top1", top1_m.average)])
    return metrics


def make_loader(
    dataset, config, batch_size: int, shuffle: bool, drop_last: bool = True
) -> DataLoader:
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle) if config.distributed else None
    )
    return DataLoader(
        dataset,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        drop_last=drop_last,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def fetch_data(config) -> tuple[Iterable, Iterable, Iterable]:
    trf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*[(0.5,), (0.5,)])]
    )

    data_dir = f"./{config.datadir}"

    # Untransformed train
    affnist_train = AffnistUntransformed(
        root=data_dir, train=True, transform=trf, download=True
    )

    # Transformed val
    # (actual eval is on transformed test)
    affnist_val = AffnistTransformed(
        root=data_dir, train=True, transform=trf, download=True
    )

    # Transformed test set
    affnist_test = AffnistTransformed(root=data_dir, train=False, transform=trf)

    train_loader = make_loader(
        affnist_train, config, batch_size=config.batch_size, shuffle=True
    )
    eval_loader = make_loader(
        affnist_val,
        config,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = make_loader(
        affnist_test,
        config,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, eval_loader, test_loader


def main():
    logging.basicConfig(level=logging.INFO)
    config = parse_args()
    configure_backends(config)
    device = init_distributed(config)
    seed_everything(config.seed)

    def loginfo(info: str) -> None:
        if is_primary(config):
            _logger.info(info)

    train_loader, eval_loader, test_loader = fetch_data(config)
    loginfo(f"train_size: {len(train_loader)}, eval_size: {len(eval_loader)}")

    model = build_model(config).to(device)
    tparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    loginfo(f"model  {model} \n ({tparams} params)")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )

    amp_autocast = nullcontext
    loss_scaler = None
    amp_dtype = torch.float16
    if config.amp:
        assert config.amp_dtype in ("float16", "bfloat16")
        if config.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

        amp_autocast = functools.partial(
            torch.autocast, device_type=device.type, dtype=amp_dtype
        )

        if device.type == "cuda" and amp_dtype == torch.float16:
            loss_scaler = NativeScaler()

    if config.distributed:
        model = DDP(model, device_ids=[config.local_rank])

    if config.torchcompile:
        model = torch.compile(model, backend=config.torchcompile)

    train_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    validate_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=config.min_lr
    )
    try:
        for epoch in range(config.max_epochs):
            if config.distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                train_loss_fn,
                config,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
            )

            if epoch and epoch % config.eval_epoch_interval == 0:
                validate(
                    epoch,
                    model,
                    eval_loader,
                    validate_loss_fn,
                    config,
                    device=device,
                    amp_autocast=amp_autocast,
                )

            if lr_scheduler is not None:
                lr_scheduler.step()

    except KeyboardInterrupt:
        pass

    loginfo("Evaluating on transformed test set.")
    test_metrics = validate(
        epoch + 1,
        model,
        test_loader,
        validate_loss_fn,
        config,
        device=device,
        amp_autocast=amp_autocast,
    )

    loginfo(f"Final test accuracy: {test_metrics['top1'] * 100:.3f}")


if __name__ == "__main__":
    main()
