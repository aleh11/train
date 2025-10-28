from cnn.core import xp, CnnLayer


def sigmoid(x):
    return 1.0 / (1.0 + xp.exp(-x))


def softmax(x, axis=-1):
    x_shifted = x - xp.max(x, axis=axis, keepdims=True) # for numerical stability
    exp_x = xp.exp(x_shifted)
    return exp_x / xp.sum(exp_x, axis=axis, keepdims=True)


class ReLU(CnnLayer):
    """ReLU activation with backward pass support."""

    def __init__(self):
        super().__init__()
        self.x_cache = None

    def forward(self, x):
        if self.training:
            self.x_cache = x.copy()  # Cache before in-place modification
        return xp.maximum(x, 0, out=x)

    def backward(self, dout):
        """Gradient passes through where input > 0."""
        return dout * (self.x_cache > 0)


class SiLU(CnnLayer):
    """SiLU (Swish) activation with backward pass support."""

    def __init__(self):
        super().__init__()
        self.x_cache = None

    def forward(self, x):
        if self.training:
            self.x_cache = x.copy()  # Cache before in-place modification
        x *= 1.0 / (1.0 + xp.exp(-x))
        return x

    def backward(self, dout):
        """Backward: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
        x = self.x_cache
        sig = 1.0 / (1.0 + xp.exp(-x))
        dx = sig + x * sig * (1.0 - sig)
        return dout * dx
