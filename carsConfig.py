# carsConfig.py — YOLOv8 training config (dataclass, no YAML)
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TrainConfig:
    # Paths
    # dataRoot: str = '/content/train/datasets/cars/'
    # outDir: str ="/content/drive/MyDrive/NOVEMBERTING"
    dataRoot: str = 'datasets/cars/' # has train/ and val/
    outDir: str = "tmp"

    # Data
    numClasses: Optional[int] = 2
    classNames: Optional[Tuple[str, ...]] = ("car", "plate")
    imageSize: int = 512  # imgsz
    batchSize: int = 64  # batch
    workers: int = 4  # workers

    # Model
    regMax: int = 16
    strides: Tuple[int, int, int] = (8, 16, 32)

    # Optimization
    epochs: int = 160  # epochs
    optimizer: str = "adamw"  # args.yaml had optimizer:auto; map to AdamW
    lr: float = 0.01
    lrf: float = 0.01
    minLr: float = 0.0001
    momentum: float = 0.937  # SGD momentum
    weightDecay: float = 5e-4  # weight_decay
    warmupEpochs: float = 3.0  # warmup_epochs
    warmupMomentum: float = 0.8  # warmup_momentum
    warmupBiasLr: float = 0.1  # warmup_bias_lr
    useCosine: bool = False  # cos_lr false
    useAmp: bool = False  # amp true
    maxGradNorm: float = 10.0

    # Early stopping / reproducibility
    patience: int = 16  # patience
    seed: int = 0  # seed
    deterministic: bool = True  # deterministic

    # Loss gains
    boxGain: float = 7.5  # box
    clsGain: float = 0.5  # cls
    dflGain: float = 1.5  # dfl

    # Assigner
    topK: int = 13
    alignAlpha: float = 1.0
    alignBeta: float = 6.0

    # Eval/export knobs you may use later
    iouThr: float = 0.7  # iou
    maxDet: int = 50  # max_det
    half: bool = False  # half

    # Fast run
    maxBatchesPerEpoch: Optional[int] = None

    # Augmentations
    mosaic: float = 1.0  # mosaic
    mixup: float = 0.0  # mixup
    hsvH: float = 0.015  # hsv_h
    hsvS: float = 0.7  # hsv_s
    hsvV: float = 0.4  # hsv_v
    flipUD: float = 0.0  # flipud
    flipLR: float = 0.5  # fliplr
    degrees: float = 0.0  # degrees
    scale: float = 0.5  # scale
    shear: float = 0.0  # shear
    translate: float = 0.1  # translate
    perspective: float = 0.0  # perspective
    bgr: float = 0.0  # bgr
    cutmix: float = 0.0  # cutmix
    copy_paste: float = 0.0  # copy_paste
    closeMosaicEpoch: int = 10  # close_mosaic
    autoAug: str = "randaugment"  # auto_augment
    randomErasing: float = 0.4  # erasing


def GetConfig() -> TrainConfig:
    return TrainConfig()
