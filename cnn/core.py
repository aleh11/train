try:
    import cupy as cp

    xp = cp
except Exception:
    import numpy as np

    xp = np


class CnnLayer:
    """
    Base class for all CNN layers. Allows loading parameters from a state dict.
    """

    def __init__(self):
        self.training = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, dout):
        """Backward pass - override in subclasses that support training."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement backward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        """Set training mode."""
        self.training = mode
        # Recursively set for child modules
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, CnnLayer):
                attr.train(mode)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, CnnLayer):
                        item.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def state_dict(self):
        """Returns dict of all parameter numpy arrays."""
        import numpy as _np

        sd= {}

        def _to_numpy(arr):
            if arr is None:
                return None
            if hasattr(xp, "asnumpy"):
                return xp.asnumpy(arr)
            if isinstance(arr, _np.ndarray):
                return arr
            return _np.array(arr)

        def walk(obj, prefix=""):
            for name, val in obj.__dict__.items():
                # Skip training mode and cache
                if name in ['training', 'cache', 'x_cache', 'mask_cache']:
                    continue

                try:
                    is_xp_array = isinstance(val, xp.ndarray)
                except Exception:
                    is_xp_array = False

                if is_xp_array or (not isinstance(val, (CnnLayer, list, dict)) and hasattr(val, "shape")):
                    # Skip gradient arrays - don't save them
                    if not name.startswith('d') and not name.endswith('_grad'):
                        sd[f"{prefix}{name}"] = _to_numpy(val)
                elif isinstance(val, CnnLayer):
                    walk(val, f"{prefix}{name}.")
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        if isinstance(item, CnnLayer):
                            walk(item, f"{prefix}{name}.{i}.")

        walk(self, "")
        return sd

    def load_state_dict(self, sd):
        for key, arr in sd.items():
            parts = key.split(".")
            obj = self

            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)

            setattr(obj, parts[-1], xp.asarray(arr, dtype=xp.float32))


class Sequential(list):
    def __init__(self, *modules):
        super().__init__(modules)
        self.training = False

    def forward(self, x):
        for m in self:
            x = m(x)
        return x

    def backward(self, dout):
        """Backward pass through sequential layers."""
        for m in reversed(self):
            if hasattr(m, 'backward'):
                dout = m.backward(dout)
        return dout

    def train(self, mode=True):
        """Set training mode for all modules."""
        self.training = mode
        for m in self:
            if hasattr(m, 'train'):
                m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    __call__ = forward


# ============================================================================
# CONVOLUTION HELPERS - im2col/col2im for efficient backward pass
# ============================================================================

def get_im2col_indices(x_shape, kernel_size, padding=0, stride=1, dilation=1):
    """
    Get indices for im2col operation.

    Returns:
        k: (C*kH*kW, 1) - channel indices
        i: (C*kH*kW, outH*outW) - height indices
        j: (C*kH*kW, outH*outW) - width indices
    """
    N, C, H, W = x_shape
    kH, kW = kernel_size

    # Account for dilation in output size calculation
    eff_kH = kH + (kH - 1) * (dilation - 1)
    eff_kW = kW + (kW - 1) * (dilation - 1)

    H_pad = H + 2 * padding
    W_pad = W + 2 * padding

    outH = (H_pad - eff_kH) // stride + 1
    outW = (W_pad - eff_kW) // stride + 1

    # Create base indices for kernel positions
    i0 = xp.repeat(xp.arange(kH), kW)
    i0 = xp.tile(i0, C)

    # Create indices for output positions
    i1 = stride * xp.repeat(xp.arange(outH), outW)

    # Width indices
    j0 = xp.tile(xp.arange(kW), kH * C)
    j1 = stride * xp.tile(xp.arange(outW), outH)

    # Combine with broadcasting
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # Apply dilation
    if dilation > 1:
        i = i * dilation
        j = j * dilation

    # Channel indices
    k = xp.repeat(xp.arange(C), kH * kW).reshape(-1, 1)

    return k, i, j


def im2col(x_padded, kernel_size, stride=1, dilation=1):
    """
    Transform input for efficient convolution.

    Args:
        x_padded: (N, C, H, W) already padded input
        kernel_size: (kH, kW)
        stride: stride value
        dilation: dilation value

    Returns:
        cols: (C*kH*kW, N*outH*outW) reshaped patches
    """
    N, C, H, W = x_padded.shape
    kH, kW = kernel_size

    # Get indices (using padding=0 since x is already padded)
    k, i, j = get_im2col_indices(x_padded.shape, kernel_size, padding=0,
                                 stride=stride, dilation=dilation)

    # Extract columns using advanced indexing
    cols = x_padded[:, k, i, j]

    # Transpose to put batch first: (N, C*kH*kW, outH*outW) -> (C*kH*kW, N*outH*outW)
    cols = cols.transpose(1, 0, 2).reshape(C * kH * kW, -1)

    return cols


def col2im(cols, x_shape, kernel_size, padding=0, stride=1, dilation=1):
    """
    Inverse of im2col - accumulates gradients.

    Args:
        cols: (C*kH*kW, N*outH*outW) gradient columns
        x_shape: (N, C, H, W) original input shape
        kernel_size: (kH, kW)
        padding: padding value
        stride: stride value
        dilation: dilation value

    Returns:
        dx: (N, C, H, W) gradient w.r.t input
    """
    N, C, H, W = x_shape
    kH, kW = kernel_size

    H_pad = H + 2 * padding
    W_pad = W + 2 * padding

    eff_kH = kH + (kH - 1) * (dilation - 1)
    eff_kW = kW + (kW - 1) * (dilation - 1)
    outH = (H_pad - eff_kH) // stride + 1
    outW = (W_pad - eff_kW) // stride + 1

    x_padded = xp.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    # Reshape cols: (C*kH*kW, N*outH*outW) -> (C, kH, kW, N, outH, outW)
    cols_reshaped = cols.reshape(C, kH, kW, N, outH, outW)

    # Loop over output positions and accumulate
    for out_h in range(outH):
        for out_w in range(outW):
            h_start = out_h * stride
            w_start = out_w * stride

            for kh in range(kH):
                for kw in range(kW):
                    h = h_start + kh * dilation
                    w = w_start + kw * dilation

                    # Accumulate gradients (transpose from (C, N) to (N, C))
                    x_padded[:, :, h, w] += cols_reshaped[:, kh, kw, :, out_h, out_w].T

    # Remove padding
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    else:
        return x_padded
