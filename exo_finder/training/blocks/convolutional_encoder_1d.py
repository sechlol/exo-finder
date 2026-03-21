from dataclasses import dataclass
from torch import nn, Tensor


@dataclass
class ConvolutionalEncoderParams:
    num_layers: int
    kernel_size: int
    in_features: int
    kaiming_initialization: bool = True
    bias: bool = False


class ConvolutionalEncoder1D(nn.Module):
    def __init__(self, params: ConvolutionalEncoderParams):
        super().__init__()
        self.params = params

        # Optional: guarantee length preservation with conv
        assert params.kernel_size % 2 == 1, "Use odd kernel_size for exact length preservation."
        layers: list[nn.Module] = []
        padding = "same"  # requires PyTorch with 'same' support; otherwise use params.kernel_size // 2 with odd kernels
        current_channels = params.in_features

        for _ in range(self.params.num_layers):
            out_channels = current_channels * 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=self.params.kernel_size,
                        padding=padding,  # or padding=self.params.kernel_size // 2 (odd kernels)
                        bias=self.params.bias,
                    ),
                    nn.BatchNorm1d(num_features=out_channels),
                    nn.SiLU(),  # Swish
                    nn.MaxPool1d(kernel_size=2, stride=2),  # floor(L/2)
                ]
            )
            current_channels = out_channels

        self.layers = nn.Sequential(*layers)

        if self.params.kaiming_initialization:
            self.layers.apply(self._init_kaiming_silu)

    @staticmethod
    def _init_kaiming_silu(m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, C_in]
        Returns:
            y: [B, floor(L / 2**num_layers), C_in * 2**num_layers]
        Note: exact length depends on pooling (floor) and convolution padding; with padding='same' it’s as above.
        """
        x = x.permute(0, 2, 1)  # [B, C_in, L]
        x = self.layers(x)  # [B, C_out, L']
        x = x.permute(0, 2, 1)  # [B, L', C_out]
        return x
