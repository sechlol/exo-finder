from torch import nn, Tensor


class DebugModule(nn.Module):
    def __init__(self, message=""):
        super(DebugModule, self).__init__()
        self._message = message

    def forward(self, x):
        if self._message:
            print(f"{self._message}: {x.shape}")
        return x


class SqueezeDim(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x.squeeze(self.dim)


class UnsqueezeDim(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class AlterShape(nn.Module):
    def __init__(self, new_shape: list[int], reshape: bool = False) -> None:
        super(AlterShape, self).__init__()
        self.new_shape = new_shape
        self.reshape = reshape

    def forward(self, x: Tensor):
        if self.reshape:
            return x.reshape(*self.new_shape)
        return x.view(*self.new_shape)


class BatchNorm1dWithPermutation(nn.Module):
    """
    BatchNorm1D wants the input tensor to be in the shape
    [BATCH_SIZE, N_FEATURES, SEQUENCE_LEN], which is different from the shape I get from the data loader:
    [BATCH_SIZE, SEQUENCE_LEN, N_FEATURES]. This class takes care of permuting the input before and after applying the
    normalization.
    """

    def __init__(self, num_features: int):
        super(BatchNorm1dWithPermutation, self).__init__()
        self._batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self._batch_norm(x)
        x = x.permute(0, 2, 1)
        return x
