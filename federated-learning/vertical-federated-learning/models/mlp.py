from torch import nn


class TwoLayerMlp(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super(TwoLayerMlp, self).__init__()

        self._input_dimensions = input_dimensions
        self._output_dimensions = output_dimensions

        self.layer = nn.Sequential(
            nn.Sequential(nn.Linear(input_dimensions, output_dimensions), nn.ReLU()),
            nn.Sequential(
                nn.Linear(output_dimensions, output_dimensions),
                nn.ReLU(),
            ),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.layer(x)

    @property
    def input_dimensions(self) -> int:
        return self._input_dimensions

    @property
    def output_dimensions(self) -> int:
        return self._output_dimensions
