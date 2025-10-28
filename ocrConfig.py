# libConfig.py — YOLOv8 training config (dataclass, no YAML)
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TrainConfig:
    # Paths
    dataRoot: str = '/content/train/datasets/plates/'
    outDir: str ="/content/drive/MyDrive/NOVEMBERTING"

    # Data
    numClasses: Optional[int] = 30  # set to your dataset
    classNames: Optional[Tuple[str]] = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
                               'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z')
    imageSize: int = 160
    batchSize: int = 16
    workers: int = 4

    # Model (YOLOv8)
    regMax: int = 16
    strides: Tuple[int, int, int] = (8, 16, 32)

    # Optimization
    epochs: int = 100
    optimizer: str = "adamw"  # "adamw" or "sgd"
    lr: float = 0.01  # lr0
    minLr: float = 0.01 * 0.01  # lrf * lr0  (lrf=0.01 by default)
    momentum: float = 0.937  # used if optimizer="sgd"
    weightDecay: float = 5e-4
    warmupEpochs: float = 3.0
    useCosine: bool = True  # cosine LR if True, else linear
    useAmp: bool = True
    maxGradNorm: float = 10.0

    # Loss gains
    boxGain: float = 7.5
    clsGain: float = 0.5
    dflGain: float = 1.5

    # Assigner (Task-Aligned)
    topK: int = 13
    alignAlpha: float = 1.0
    alignBeta: float = 6.0

    # Fast run knob
    maxBatchesPerEpoch: Optional[int] = None

    # Augmentations (Ultralytics-style)
    mosaic: float = 1.0
    mixup: float = 0.15
    hsvH: float = 0.015
    hsvS: float = 0.7
    hsvV: float = 0.4
    flipUD: float = 0.0
    flipLR: float = 0.5
    degrees: float = 0.0
    scale: float = 0.9
    shear: float = 0.0
    translate: float = 0.1


def GetConfig() -> TrainConfig:
    return TrainConfig()
