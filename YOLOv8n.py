import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Conv(nn.Module):
    """
    Convolution + BatchNorm2d + SiLU activation block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=True, eps=0.001, momentum=0.03, track_running_stats=True)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Residual bottleneck block: two Conv layers with optional skip connection.
    """

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.cv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut and (in_channels == out_channels)

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return out + x if self.shortcut else out


class C2f(nn.Module):
    """
    Cross Stage Partial (CSP)-like block: split, bottleneck sequence, and merge.
    """

    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        self.midChannels = out_channels // 2
        self.cv2OutChannels = (n + 2) * self.midChannels

        self.cv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.cv2 = Conv(self.cv2OutChannels, out_channels, kernel_size=1, stride=1, padding=0)

        self.m = nn.ModuleList([
            Bottleneck(self.midChannels, self.midChannels, shortcut) for _ in range(n)
        ])

    def forward(self, x):
        x = self.cv1(x)
        y = list(x.chunk(2, 1))  # Create list with [x1, x2] like Ultralytics
        y.extend(m(y[-1]) for m in self.m)  # Extend with bottleneck outputs
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast variant: pools with kernel=5 three times.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hiddenChannels = out_channels // 2
        self.cv1 = Conv(in_channels, self.hiddenChannels, 1, 1, 0)

        self.cv2 = Conv(4 * self.hiddenChannels, out_channels, 1, 1, 0)

        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.

    This is the key innovation in YOLOv8. Instead of directly predicting bbox coordinates,
    we predict probability distributions over possible distances, then convert these
    distributions to actual distance values.

    Think of it like this: instead of saying "the object boundary is exactly 3.7 pixels away",
    we say "there's a 60% chance it's 3 pixels, 30% chance it's 4 pixels, 10% chance it's 5 pixels"
    and then compute the weighted average.
    """

    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        # Create a 1x1 conv that acts as a weighted sum
        # The weights are just [0, 1, 2, 3, ..., c1-1] which represent distance values
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)

        # Initialize the weights as sequential integers
        weight = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.conv.weight.data = weight

    def forward(self, x):
        """
        Convert distribution predictions to distance values.

        Args:
            x: Tensor of shape (batch, 4*c1, height, width) - the raw distribution logits

        Returns:
            Tensor of shape (batch, 4, height, width) - the distance predictions
        """
        b, c, h, w = x.shape
        # Reshape to separate the 4 directions and c1 distribution bins
        # (b, 4*c1, h, w) -> (b, 4, c1, h, w)
        x = x.view(b, 4, self.c1, h, w)

        # Apply softmax to convert logits to probabilities along the c1 dimension
        # This ensures each distribution sums to 1
        x = F.softmax(x, dim=2)

        # Reshape for the convolution: (b, 4, c1, h, w) -> (b*4, c1, h, w)
        x = x.view(b * 4, self.c1, h, w)

        # Apply the weighted sum convolution to get distance values
        x = self.conv(x)

        # Reshape back: (b*4, 1, h, w) -> (b, 4, h, w)
        return x.view(b, 4, h, w)


class Detect(nn.Module):
    """
    YOLOv8 Detection Head - Anchor-free with Distribution Focal Loss

    This implementation matches the official Ultralytics code exactly.
    Key features:
    1. Anchor-free: No predefined anchor boxes needed
    2. Split head: Separate branches for bbox regression and classification
    3. DFL: Uses distribution predictions for more stable bbox regression
    4. Multi-scale: Processes features from multiple pyramid levels
    """

    # Class variables for state management
    dynamic = False  # Force grid reconstruction
    export = False  # Export mode flag
    shape = None  # Input shape cache
    anchors = torch.empty(0)  # Anchor points cache
    strides = torch.empty(0)  # Stride values cache

    def __init__(self, nc=80, ch=()):
        """
        Initialize the detection head.

        Args:
            nc (int): Number of classes (default 80 for COCO dataset)
            ch (tuple): Input channel dimensions from different FPN levels
                       Typically (256, 512, 1024) for YOLOv8n or similar
        """
        super().__init__()

        # Store basic configuration
        self.nc = nc  # Number of classes
        self.nl = len(ch)  # Number of detection layers (typically 3)
        self.reg_max = 16  # Number of bins in each distance distribution
        self.no = nc + self.reg_max * 4  # Total outputs per anchor point
        self.stride = torch.zeros(self.nl)  # Will be set during model build

        # Calculate channel dimensions for the two branches
        # c2: Regression branch channels - enough for complex bbox prediction
        # c3: Classification branch channels - enough for class discrimination
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))  # Cap at 100 to prevent excessive parameters

        # Build regression branch (predicts bbox coordinates via distributions)
        # Each branch: input -> c2 -> c2 -> 4*reg_max outputs
        # The 4*reg_max outputs represent 4 distance distributions (left, top, right, bottom)
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                Conv(x, c2, 3),  # 3x3 conv with activation
                Conv(c2, c2, 3),  # Another 3x3 conv with activation
                nn.Conv2d(c2, 4 * self.reg_max, 1)  # Final 1x1 conv to get distributions
            ) for x in ch
        ])

        # Build classification branch (predicts class probabilities)
        # Each branch: input -> c3 -> c3 -> nc outputs
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                Conv(x, c3, 3),  # 3x3 conv with activation
                Conv(c3, c3, 3),  # Another 3x3 conv with activation
                nn.Conv2d(c3, self.nc, 1)  # Final 1x1 conv to get class logits
            ) for x in ch
        ])

        # Initialize the DFL module for converting distributions to distances
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the detection head.

        Args:
            x (list): List of feature tensors from different FPN levels
                     Each tensor has shape (batch, channels, height, width)

        Returns:
            During training: List of tensors with combined bbox+class predictions
            During inference: Single tensor with final detections
        """
        shape = x[0].shape  # Get batch size and spatial dimensions

        # Process each feature level through both branches
        predictions = []
        for i in range(self.nl):
            # Get bbox distribution predictions: (batch, 4*reg_max, h, w)
            bbox_dist = self.cv2[i](x[i])

            # Get class predictions: (batch, nc, h, w)
            cls_pred = self.cv3[i](x[i])

            # Combine predictions: (batch, 4*reg_max + nc, h, w)
            combined = torch.cat([bbox_dist, cls_pred], dim=1)
            predictions.append(combined)

        # During training, return the raw predictions for loss calculation
        if self.training:
            return predictions

        # During inference, we need to convert to final detection format
        return self.inference(predictions, shape)

    def inference(self, predictions, shape):
        """
        Convert training predictions to final inference format.

        This handles:
        1. Converting distribution predictions to actual bbox coordinates
        2. Applying sigmoid to class predictions
        3. Scaling coordinates by stride values
        4. Concatenating all detection levels

        Args:
            predictions (list): Raw predictions from forward pass
            shape (tuple): Input tensor shape for caching anchors

        Returns:
            torch.Tensor: Final detections (batch, 4+nc, total_anchors)
        """
        # Generate anchor points if needed (cached for efficiency)
        if self.shape != shape:
            self.anchors, self.strides = self._make_anchors(predictions, self.stride, 0.5)
            self.shape = shape

        # Process each prediction level separately before concatenating
        # This is the key fix - we need to apply DFL before flattening spatial dimensions
        processed_predictions = []

        for i, pred in enumerate(predictions):
            # Split into bbox distributions and class logits for this level
            # pred shape: (batch, 4*reg_max + nc, height, width)
            bbox_dist = pred[:, :self.reg_max * 4]  # (batch, 4*reg_max, h, w)
            cls_logits = pred[:, self.reg_max * 4:]  # (batch, nc, h, w)

            # Apply DFL to convert distributions to distances
            # This needs to happen while we still have spatial dimensions intact
            distances = self.dfl(bbox_dist)  # (batch, 4, h, w)

            # Apply sigmoid to class predictions
            cls_probs = cls_logits.sigmoid()  # (batch, nc, h, w)

            # Combine distances and class probabilities
            level_output = torch.cat([distances, cls_probs], dim=1)  # (batch, 4+nc, h, w)
            processed_predictions.append(level_output)

        # Now flatten all predictions and concatenate
        x_cat = torch.cat([pred.view(shape[0], 4 + self.nc, -1) for pred in processed_predictions], 2)

        # Split into final bbox coordinates and class probabilities
        bbox, cls = x_cat.split((4, self.nc), 1)

        # Convert distances to actual bbox coordinates using anchor points
        bbox = self._dist2bbox(bbox, self.anchors.unsqueeze(0), xywh=True, dim=1)

        # Scale by stride to get coordinates in original image space
        # bbox has shape (batch, 4, total_anchors)
        # self.strides has shape (total_anchors, 1)
        # We need to reshape strides to be broadcastable: (1, 1, total_anchors)
        strides_broadcast = self.strides.T.unsqueeze(0)  # (total_anchors, 1) -> (1, total_anchors) -> (1, 1, total_anchors)
        bbox *= strides_broadcast

        # Combine final bbox coordinates with class probabilities
        result = torch.cat([bbox, cls], 1)
        return result

    def _make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """
        Generate anchor points and stride tensors for all feature levels.

        In anchor-free detection, we still need reference points (anchors) for each
        grid cell to convert relative predictions to absolute coordinates.

        Args:
            feats (list): Feature tensors from different levels
            strides (torch.Tensor): Stride values for each level
            grid_cell_offset (float): Offset for grid cell centers (0.5 = center)

        Returns:
            tuple: (anchor_points, stride_tensor)
        """
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device

        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            # Create grid coordinates for this feature level
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')

            # Stack x,y coordinates and add offset to get cell centers
            grid_xy = torch.stack([grid_x, grid_y], -1) + grid_cell_offset

            # Flatten to get all anchor points for this level: (h*w, 2)
            anchor_points.append(grid_xy.view(-1, 2))

            # Create stride tensor with same number of points: (h*w, 1)
            stride_tensor.append(torch.full((h * w, 1), strides[i], dtype=dtype, device=device))

        # Concatenate all levels: (total_anchors, 2) and (total_anchors, 1)
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def _dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """
        Convert distance predictions to bounding box coordinates.

        The model predicts distances from each anchor point to the object boundaries
        in 4 directions: left, top, right, bottom. This function converts those
        distances into standard bbox coordinates.

        Args:
            distance (torch.Tensor): Distance predictions (batch, 4, anchors)
            anchor_points (torch.Tensor): Grid anchor points (batch, anchors, 2)
            xywh (bool): If True, return (x_center, y_center, width, height)
                        If False, return (x1, y1, x2, y2)
            dim (int): Dimension along which to split distance predictions

        Returns:
            torch.Tensor: Bounding box coordinates (batch, 4, anchors)
        """
        # Transpose distance from (batch, 4, anchors) to (batch, anchors, 4) for easier processing
        distance = distance.transpose(-1, -2)  # (batch, anchors, 4)

        # Split distance predictions into left-top and right-bottom pairs
        # distance has shape (batch, anchors, 4) where the 4 dimensions are [left, top, right, bottom]
        lt = distance[..., :2]  # left, top distances (batch, anchors, 2)
        rb = distance[..., 2:]  # right, bottom distances (batch, anchors, 2)

        # Convert distances to actual coordinates
        # anchor_points has shape (batch, anchors, 2) representing (x, y) coordinates
        # lt contains (left, top) distances, rb contains (right, bottom) distances
        x1y1 = anchor_points - lt  # Top-left corner: anchor - (left_dist, top_dist)
        x2y2 = anchor_points + rb  # Bottom-right corner: anchor + (right_dist, bottom_dist)

        if xywh:
            # Convert to center coordinates and width/height
            c_xy = (x1y1 + x2y2) / 2  # Center point (batch, anchors, 2)
            wh = x2y2 - x1y1  # Width and height (batch, anchors, 2)

            # Combine center and size: (batch, anchors, 4)
            result = torch.cat([c_xy, wh], dim=-1)

            # Transpose back to (batch, 4, anchors) to match expected output format
            result = result.transpose(-1, -2)
            return result

        # Return corner coordinates (batch, anchors, 4) then transpose to (batch, 4, anchors)
        result = torch.cat([x1y1, x2y2], dim=-1)
        return result.transpose(-1, -2)

    def bias_init(self):
        """
        Initialize detection head biases for better training convergence.

        This is called after the model is built and stride values are available.
        The bias initialization helps the model start training with reasonable
        predictions rather than random outputs.
        """
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            # Initialize regression branch bias to 1.0
            # This encourages the model to predict some positive distance initially
            a[-1].bias.data[:] = 1.0

            # Initialize classification branch bias based on expected object frequency
            # This formula assumes roughly 5/num_classes objects per 640x640 image
            # The bias makes rare classes less likely to be predicted initially
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLOv8Nano(nn.Module):
    """
    YOLOv8-nano architecture.
    """

    def __init__(self, num_classes=1, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        # backbone
        self.b0 = Conv(3, 16, 3, 2, 1)
        self.b1 = Conv(16, 32, 3, 2, 1)
        self.b2 = C2f(32, 32, n=1)
        self.b3 = Conv(32, 64, 3, 2, 1)
        self.b4 = C2f(64, 64, n=2)
        self.b5 = Conv(64, 128, 3, 2, 1)
        self.b6 = C2f(128, 128, n=2)
        self.b7 = Conv(128, 256, 3, 2, 1)
        self.b8 = C2f(256, 256, n=1)
        self.b9 = SPPF(256, 256)
        # neck
        self.b10 = nn.Upsample(scale_factor=2, mode='nearest')
        self.b11 = Concat(dim=1)
        self.b12 = C2f(256 + 128, 128, n=1, shortcut=False)
        self.b13 = nn.Upsample(scale_factor=2, mode='nearest')
        self.b14 = Concat(dim=1)
        self.b15 = C2f(128 + 64, 64, n=1, shortcut=False)
        self.b16 = Conv(64, 64, 3, 2, 1)
        self.b17 = Concat(dim=1)
        self.b18 = C2f(128 + 64, 128, n=1, shortcut=False)
        self.b19 = Conv(128, 128, 3, 2, 1)
        self.b20 = Concat(dim=1)
        self.b21 = C2f(256 + 128, 256, n=1, shortcut=False)
        # head
        # self.b22 = DetectHead([64, 128, 256], num_classes, reg_max)
        self.b22 = Detect(nc=num_classes, ch=(64, 128, 256))

    def forward(self, x):
        # backbone
        x0 = self.b0(x)  # block  0
        x1 = self.b1(x0)  # block  1
        x1 = self.b2(x1)  # block  2
        x2 = self.b3(x1)  # block  3
        x2 = self.b4(x2)  # block  4
        x3 = self.b5(x2)  # block  5
        x3 = self.b6(x3)  # block  6
        x4 = self.b7(x3)  # block  7
        x4 = self.b8(x4)  # block  8
        x4 = self.b9(x4)  # block  9

        # neck fusion
        u1 = self.b10(x4)  # block 10 (Upsample)
        cat1 = self.b11([u1, x3])  # block 11 (Concat)
        f1 = self.b12(cat1)  # block 12 (C2F)

        u2 = self.b13(f1)  # block 13 (Upsample)
        cat2 = self.b14([u2, x2])  # block 14 (Concat)
        d0 = self.b15(cat2)  # block 15 (C2F)

        d1_in = self.b16(d0)  # block 16 (Conv)
        cat3 = self.b17([d1_in, f1])  # block 17 (Concat)
        d1 = self.b18(cat3)  # block 18 (C2F)

        d2_in = self.b19(d1)  # block 19 (Conv)
        cat4 = self.b20([d2_in, x4])  # block 20 (Concat)
        d2 = self.b21(cat4)  # block 21 (C2F)

        # head
        return self.b22([d0, d1, d2])  # block 22 (DetectHead)
