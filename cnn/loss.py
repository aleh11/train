import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from cnn.core import xp
from cnn.activations import softmax
from cnn.utils import make_anchors

class CrossEntropyLoss:
    """Cross entropy loss with optional label smoothing."""

    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing
        self.probs = None

    def forward(self, logits, targets):
        """Compute cross entropy loss."""
        batch_size, num_classes = logits.shape

        # Softmax
        self.probs = softmax(logits, axis=1)

        # One-hot encode targets
        targets_oh = xp.zeros((batch_size, num_classes), dtype=xp.float32)
        targets_oh[xp.arange(batch_size), targets] = 1.0

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_oh = targets_oh * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        # Compute loss
        log_probs = xp.log(self.probs + 1e-8)
        loss = -xp.sum(targets_oh * log_probs) / batch_size

        return loss

    def backward(self, targets):
        """Compute gradient of cross entropy w.r.t logits."""
        batch_size, num_classes = self.probs.shape

        # One-hot encode targets
        targets_oh = xp.zeros((batch_size, num_classes), dtype=xp.float32)
        targets_oh[xp.arange(batch_size), targets] = 1.0

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_oh = targets_oh * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        return (self.probs - targets_oh) / batch_size


def compute_iou(boxes1, boxes2, eps=1e-7):
    """
    Element-wise IoU computation between two sets of boxes of same shape.
    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (N, 4) in xyxy format
    Returns:
        iou: (N,)
    """
    (a1, a2), (b1, b2) = boxes1.chunk(2, 1), boxes2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(1)
    return intersection / ((a2 - a1).prod(1) + (b2 - b1).prod(1) - intersection + eps)


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """
    Select positive anchor centers within ground truth boxes.
    Args:
        xy_centers: (h*w, 2) anchor center coordinates
        gt_bboxes: (b, n_max_boxes, 4) ground truth boxes in xyxy format
    Returns:
        mask_in_gts: (b, n_max_boxes, h*w) boolean mask
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape

    # Expand dimensions for broadcasting
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # (b*n_boxes, 1, 2)
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2)
    bbox_deltas = bbox_deltas.view(bs, n_boxes, n_anchors, -1)

    # Check if anchor centers are inside boxes
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """
    Select only the highest IoU prediction for each ground truth box.
    Args:
        mask_pos: (b, n_max_boxes, h*w) boolean mask of positive samples
        overlaps: (b, n_max_boxes, h*w) IoU values
        n_max_boxes: int
    Returns:
        mask_pos: (b, n_max_boxes, h*w) updated mask with only highest overlaps
        fg_mask: (b, h*w) foreground mask
    """
    # Count how many GTs each anchor is assigned to
    fg_mask = mask_pos.sum(-2)

    if fg_mask.max() > 1:  # If any anchor is assigned to multiple GTs
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        max_overlaps_idx = overlaps.argmax(1)

        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
        fg_mask = mask_pos.sum(-2)

    return mask_pos, fg_mask


class Assigner(nn.Module):
    """
    Task-aligned assigner for YOLO models.
    Assigns ground truth boxes to anchor points based on alignment metric.
    """

    def __init__(self, nc=80, top_k=13, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.nc = nc
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """
        Assign ground truths to predictions.

        Args:
            pred_scores: (b, h*w, nc) predicted class scores
            pred_bboxes: (b, h*w, 4) predicted boxes in xyxy format
            anchor_points: (h*w, 2) anchor point coordinates
            gt_labels: (b, n_max_boxes, 1) ground truth class labels
            gt_bboxes: (b, n_max_boxes, 4) ground truth boxes in xyxy format
            mask_gt: (b, n_max_boxes, 1) mask for valid ground truths

        Returns:
            target_bboxes: (b, h*w, 4) assigned target boxes
            target_scores: (b, h*w, nc) assigned target scores
            fg_mask: (b, h*w) foreground mask
        """
        self.bs = pred_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pred_scores[..., 0], 0).unsqueeze(-1),
                torch.zeros_like(pred_scores),
                torch.zeros_like(pred_scores[..., 0]) > 0
            )

        # Get anchors inside ground truth boxes
        mask_in_gts = select_candidates_in_gts(anchor_points, gt_bboxes)

        # Compute alignment metric
        align_metric, overlaps = self.get_box_metrics(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )

        # Select top-k candidates
        mask_top_k = self.select_topk_candidates(align_metric, top_k_mask=mask_gt.expand(-1, -1, self.top_k).bool())

        # Merge all masks
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        # Handle overlapping assignments
        mask_pos, fg_mask = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assign targets
        target_gt_idx = mask_pos.argmax(-2)

        # Get batch indices
        batch_idx = torch.arange(self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_idx = target_gt_idx + batch_idx * self.n_max_boxes

        # Assign labels
        target_labels = gt_labels.long().flatten()[target_idx]

        # Assign bboxes
        target_bboxes = gt_bboxes.view(-1, 4)[target_idx]

        # Clamp labels to valid range
        target_labels.clamp_(0)

        # Create one-hot target scores
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.nc),
            dtype=torch.int64,
            device=target_labels.device
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        # Mask background
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize by alignment metric
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()

    def get_box_metrics(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric and overlaps."""
        na = pred_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pred_bboxes.dtype, device=pred_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pred_scores.dtype, device=pred_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)

        bbox_scores[mask_gt] = pred_scores[ind[0], :, ind[1]][mask_gt]

        # Compute IoU - FIXED: element-wise IoU for aligned boxes
        pd_boxes = pred_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = compute_iou(gt_boxes, pd_boxes).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, top_k_mask=None):
        """Select top-k candidates based on metrics."""
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=largest)

        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)

        top_k_indices.masked_fill_(~top_k_mask, 0)

        # Create mask
        mask = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8)

        for k in range(self.top_k):
            mask.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)

        mask.masked_fill_(mask > 1, 0)
        return mask.to(metrics.dtype)


class BoxLoss(torch.nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.dfl_ch - 0.01)
        loss_dfl = self.df_loss(pred_dist[fg_mask].view(-1, self.dfl_ch + 1), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_box, loss_dfl

    @staticmethod
    def df_loss(pred_dist, target):
        # Distribution Focal Loss (DFL)
        # https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        left_loss = cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


class ComputeLoss:
    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device

        # Support both naming conventions
        m = model.b22

        self.params = params
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        # Fix: use reg_max instead of ch
        self.reg_max = m.reg_max
        self.device = device

        self.box_loss = BoxLoss(m.reg_max - 1).to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)

        self.project = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def box_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4)
        pred_dist = pred_dist.softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return torch.cat(tensors=(x1y1, x2y2), dim=-1)

    def __call__(self, outputs, targets):
        x = torch.cat([i.view(outputs[0].shape[0], self.no, -1) for i in outputs], dim=2)
        pred_distri, pred_scores = x.split(split_size=(self.reg_max * 4, self.nc), dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        data_type = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        input_size = torch.tensor(outputs[0].shape[2:], device=self.device, dtype=data_type) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)

        idx = targets['idx'].view(-1, 1)
        cls = targets['cls'].view(-1, 1)
        box = targets['box']

        targets = torch.cat((idx, cls, box), dim=1).to(self.device)
        if targets.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
            y = torch.empty_like(x)
            dw = x[..., 2] / 2  # half-width
            dh = x[..., 3] / 2  # half-height
            y[..., 0] = x[..., 0] - dw  # top left x
            y[..., 1] = x[..., 1] - dh  # top left y
            y[..., 2] = x[..., 0] + dw  # bottom right x
            y[..., 3] = x[..., 1] + dh  # bottom right y
            gt[..., 1:5] = y
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchor_points, pred_distri)
        assigned_targets = self.assigner(pred_scores.detach().sigmoid(),
                                         (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                         anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask = assigned_targets

        target_scores_sum = max(target_scores.sum(), 1)

        loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum  # BCE

        # Box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_box, loss_dfl = self.box_loss(pred_distri,
                                               pred_bboxes,
                                               anchor_points,
                                               target_bboxes,
                                               target_scores,
                                               target_scores_sum, fg_mask)

        loss_box *= self.params['box']  # box gain
        loss_cls *= self.params['cls']  # cls gain
        loss_dfl *= self.params['dfl']  # dfl gain

        return loss_box, loss_cls, loss_dfl