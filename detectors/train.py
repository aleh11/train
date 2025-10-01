# train_yolo.py
import os
import time
import random
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from YOLOv8n import YOLOv8Nano
from dataset import Dataset  # Using the big dataset file with mosaic, mixup, etc.
from utils import (
    ComputeLoss, EMA, CosineLR, setup_seed,
    non_max_suppression, compute_metric, compute_ap,
    clip_gradients, AverageMeter
)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg, ema=None):
    """Train for one epoch"""
    model.train()

    loss_meter = AverageMeter()
    box_meter = AverageMeter()
    cls_meter = AverageMeter()
    dfl_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", unit="batch")

    max_batches = getattr(cfg, 'maxBatchesPerEpoch', None)

    for batch_idx, (imgs, targets) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        imgs = imgs.to(device, non_blocking=True)
        # targets is already a dict with 'cls', 'box', 'idx' keys
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        bs = imgs.size(0)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with autocast(enabled=cfg.useAmp and device.type == 'cuda'):
            outputs = model(imgs)
            loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
            loss = loss_box + loss_cls + loss_dfl

        # Backward pass
        if cfg.useAmp and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradients(model, max_norm=cfg.maxGradNorm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_gradients(model, max_norm=cfg.maxGradNorm)
            optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        # Update metrics
        loss_meter.update(loss.item(), bs)
        box_meter.update(loss_box.item(), bs)
        cls_meter.update(loss_cls.item(), bs)
        dfl_meter.update(loss_dfl.item(), bs)

        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.3f}",
            'box': f"{box_meter.avg:.3f}",
            'cls': f"{cls_meter.avg:.3f}",
            'dfl': f"{dfl_meter.avg:.3f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    return {
        'loss': loss_meter.avg,
        'box_loss': box_meter.avg,
        'cls_loss': cls_meter.avg,
        'dfl_loss': dfl_meter.avg
    }


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, cfg):
    """Validate the model"""
    model.eval()

    loss_meter = AverageMeter()
    box_meter = AverageMeter()
    cls_meter = AverageMeter()
    dfl_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Valid Epoch {epoch:03d}", unit="batch")

    # For mAP computation
    all_stats = []
    iou_v = torch.linspace(0.5, 0.95, 10, device=device)

    for imgs, targets in pbar:
        imgs = imgs.float() / 255.0
        imgs = imgs.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        bs = imgs.size(0)

        # IMPORTANT: Set to train mode for loss computation
        model.train()
        outputs = model(imgs)
        loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
        loss = loss_box + loss_cls + loss_dfl

        # Switch back to eval mode for NMS
        model.eval()
        # Get predictions for mAP (need to run forward again in eval mode)
        with torch.no_grad():
            preds_for_nms = model(imgs)

        conf_thresh = getattr(cfg, 'confThresh', 0.001)
        iou_thresh = getattr(cfg, 'iouThr', 0.7)

        preds = non_max_suppression(
            preds_for_nms,
            confidence_threshold=conf_thresh,
            iou_threshold=iou_thresh
        )

        # Update metrics
        loss_meter.update(loss.item(), bs)
        box_meter.update(loss_box.item(), bs)
        cls_meter.update(loss_cls.item(), bs)
        dfl_meter.update(loss_dfl.item(), bs)

        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.3f}",
            'box': f"{box_meter.avg:.3f}",
            'cls': f"{cls_meter.avg:.3f}",
            'dfl': f"{dfl_meter.avg:.3f}"
        })

    return {
        'loss': loss_meter.avg,
        'box_loss': box_meter.avg,
        'cls_loss': cls_meter.avg,
        'dfl_loss': dfl_meter.avg
    }


def main(cfg):
    setup_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    print(f"Using device: {device}")

    # Create directories
    out_dir = Path(cfg.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = out_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    print("Building datasets...")
    print(f"Data root: {cfg.dataRoot}")

    # Get filenames from directory structure: train/images/*.jpg
    FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')

    train_img_dir = Path(cfg.dataRoot) / 'train' / 'images'
    val_img_dir = Path(cfg.dataRoot) / 'val' / 'images'

    train_filenames = [str(f) for f in train_img_dir.iterdir()
                       if f.suffix[1:].lower() in FORMATS]
    val_filenames = [str(f) for f in val_img_dir.iterdir()
                     if f.suffix[1:].lower() in FORMATS]

    print(f"Found {len(train_filenames)} training images")
    print(f"Found {len(val_filenames)} validation images")

    # Augmentation params from config
    aug_params = {
        'mosaic': cfg.mosaic,
        'mix_up': cfg.mixup,
        'degrees': cfg.degrees,
        'translate': cfg.translate,
        'scale': cfg.scale,
        'shear': cfg.shear,
        'flip_ud': cfg.flipUD,
        'flip_lr': cfg.flipLR,
        'hsv_h': cfg.hsvH,
        'hsv_s': cfg.hsvS,
        'hsv_v': cfg.hsvV
    }

    train_dataset = Dataset(
        filenames=train_filenames,
        input_size=cfg.imageSize,
        params=aug_params,
        augment=True
    )

    val_dataset = Dataset(
        filenames=val_filenames,
        input_size=cfg.imageSize,
        params=aug_params,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batchSize,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batchSize,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )

    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Build model
    print("Building model...")
    model = YOLOv8Nano(num_classes=cfg.numClasses, reg_max=cfg.regMax)
    model = model.to(device)

    # Initialize strides and biases
    model.b22.stride = torch.tensor(cfg.strides, dtype=torch.float32)
    model.b22.bias_init()

    # Build loss function
    loss_params = {
        'box': cfg.boxGain,
        'cls': cfg.clsGain,
        'dfl': cfg.dflGain
    }
    criterion = ComputeLoss(model, loss_params)

    # Build optimizer
    if cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weightDecay
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weightDecay,
            nesterov=True
        )

    # Build scheduler
    num_steps = len(train_loader)
    scheduler_params = {
        'max_lr': cfg.lr,
        'min_lr': cfg.minLr,
        'warmup_epochs': cfg.warmupEpochs
    }

    class Args:
        epochs = cfg.epochs

    scheduler = CosineLR(Args(), scheduler_params, num_steps)

    # EMA
    use_ema = getattr(cfg, 'useEma', True)  # Default to True if not in config
    ema_decay = getattr(cfg, 'emaDecay', 0.9999)
    ema = EMA(model, decay=ema_decay) if use_ema else None

    # Mixed precision
    scaler = GradScaler(enabled=(cfg.useAmp and device.type == 'cuda'))

    # Resume from checkpoint
    start_epoch = 1
    best_loss = float('inf')
    last_ckpt = out_dir / 'last.pt'

    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema and 'ema' in ckpt:
            ema.ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch - 1}")

    # CSV logging
    log_path = stats_dir / f"yolov8n_{cfg.epochs}e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,train_box,train_cls,train_dfl,'
                'val_loss,val_box,val_cls,val_dfl,lr,time_sec,best_loss\n')

    # Training loop
    print(f"\nStarting training for {cfg.epochs} epochs...")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, epoch, cfg, ema
        )

        # Validate
        val_model = ema.ema if ema else model
        val_metrics = validate(
            val_model, val_loader, criterion,
            device, epoch, cfg
        )

        # Step scheduler
        for _ in range(num_steps):
            step = (_ + num_steps * (epoch - 1))
            scheduler.step(step, optimizer)

        elapsed = time.time() - t0

        # Print summary
        print(f"\n[Epoch {epoch:03d}] "
              f"Train: loss={train_metrics['loss']:.4f} "
              f"box={train_metrics['box_loss']:.4f} "
              f"cls={train_metrics['cls_loss']:.4f} "
              f"dfl={train_metrics['dfl_loss']:.4f} | "
              f"Val: loss={val_metrics['loss']:.4f} "
              f"box={val_metrics['box_loss']:.4f} "
              f"cls={val_metrics['cls_loss']:.4f} "
              f"dfl={val_metrics['dfl_loss']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"{elapsed:.1f}s\n")

        # Log to CSV
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_metrics['loss']:.6f},"
                    f"{train_metrics['box_loss']:.6f},"
                    f"{train_metrics['cls_loss']:.6f},"
                    f"{train_metrics['dfl_loss']:.6f},"
                    f"{val_metrics['loss']:.6f},"
                    f"{val_metrics['box_loss']:.6f},"
                    f"{val_metrics['cls_loss']:.6f},"
                    f"{val_metrics['dfl_loss']:.6f},"
                    f"{optimizer.param_groups[0]['lr']:.8f},"
                    f"{elapsed:.2f},{best_loss:.6f}\n")

        # Save checkpoints
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'cfg': vars(cfg),
            'best_loss': best_loss
        }
        if ema:
            ckpt['ema'] = ema.ema.state_dict()

        torch.save(ckpt, out_dir / 'last.pt')

        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save(ckpt, out_dir / 'best.pt')
            print(f"✓ Saved best model (loss={best_loss:.4f})")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}")
    print(f"Logs saved to: {log_path}")


if __name__ == '__main__':
    import ocrConfig, carsConfig

    carsCfg = carsConfig.GetConfig()
    ocrCfg = ocrConfig.GetConfig()
    cars = True
    main(carsCfg if cars else ocrCfg)
