import numpy
import numpy as np
import torch


def compute_metric(output, target, iou_v):
    """
    Compute correctness matrix for predictions.

    Args:
        output: (N, 6) tensor [x1, y1, x2, y2, conf, cls]
        target: (M, 5) tensor [cls, x1, y1, x2, y2]
        iou_v: (K,) tensor of IoU thresholds

    Returns:
        correct: (N, K) boolean tensor indicating correct predictions
    """
    # Compute intersection
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # Compute IoU
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)

    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def smooth(y, f=0.05):
    """
    Box filter smoothing.

    Args:
        y: array to smooth
        f: fraction of array length to use for filter width

    Returns:
        smoothed array
    """
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

    Args:
        tp: True positives (nparray, nx1 or nx10)
        conf: Object-ness value from 0-1 (nparray)
        pred_cls: Predicted object classes (nparray)
        target_cls: True object classes (nparray)
        eps: Small value to prevent division by zero

    Returns:
        tp: True positives at best F1 threshold
        fp: False positives at best F1 threshold
        m_pre: Mean precision
        m_rec: Mean recall
        map50: AP@0.5
        mean_ap: AP@0.5:0.95
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px = numpy.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()

    return tp, fp, m_pre, m_rec, map50, mean_ap


def accuracy_topk(logits, targets, topk=(1,)):
    """Compute top-k accuracy."""
    batch_size = logits.shape[0]
    maxk = max(topk)

    # Get top-k predictions
    top_pred = np.argsort(logits, axis=1)[:, ::-1][:, :maxk]

    # Check correctness
    correct = top_pred == targets[:, None]

    # Compute accuracy for each k
    res = []
    for k in topk:
        correct_k = correct[:, :k].any(axis=1).sum()
        res.append(100.0 * correct_k / batch_size)

    return res
