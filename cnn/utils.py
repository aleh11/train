import random
from time import time
from cnn.core import xp
import numpy as xp
import torch
import torchvision


def setup_seed():
    """Setup random seed for reproducibility."""
    random.seed(0)
    xp.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else xp.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def wh2xy_n2p(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = xp.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh_p2n(x, w, h):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = xp.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def make_anchors(x, strides, offset=0.5):
    """
    Generate anchor points and stride tensors for multi-scale feature maps.

    Args:
        x: list of feature tensors from different scales
        strides: tuple/list of stride values for each scale
        offset: offset for grid cell centers (default 0.5 for center)

    Returns:
        anchor_points: (total_anchors, 2) tensor of anchor coordinates
        stride_tensor: (total_anchors, 1) tensor of stride values
    """
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_tensor), torch.cat(stride_tensor)


def nmsTorch(outputs, confidence_threshold=0.001, iou_threshold=0.7):
    """
    Perform Non-Maximum Suppression (NMS) on detection outputs.

    Args:
        outputs: (batch, num_classes+4, total_anchors) tensor of predictions
        confidence_threshold: minimum confidence to keep detections
        iou_threshold: IoU threshold for NMS

    Returns:
        output: list of (N, 6) tensors [x1, y1, x2, y2, conf, cls] per image
    """
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]
    nc = outputs.shape[1] - 4
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold

    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after
    output = [torch.zeros((0, 6), device=outputs.device)] * bs

    for index, x in enumerate(outputs):
        x = x.transpose(0, -1)[xc[index]]

        if not x.shape[0]:
            continue

        # Split into box and class predictions
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)

        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

        n = x.shape[0]
        if not n:
            continue

        # Sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        indices = indices[:max_det]

        output[index] = x[indices]
        if (time() - start) > limit:
            break

    return output


def clip_gradients(model, max_norm=10.0):
    """Clip gradients by norm."""
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
