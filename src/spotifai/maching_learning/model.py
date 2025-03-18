import json
import torch

from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Conv1d,
    TransformerEncoderLayer,
    Linear,
    GELU,
    ReLU,
    BatchNorm1d,
)
from torch.nn.functional import relu


# TODO: instead of linear last layer with dim 1, try adding CLS token and use the output from that one as final feature => variable length sequences can be processed.
# TODO: remove output batch norm - perform the operation in the loss function instead.
class Model(Module):
    """Audio encoder neural network. Inspired by Whisper [https://github.com/openai/whisper].

    The computation chain of the encoder is:

    1. Convolutional (1D) encoding over mel bins (feature dimension).
    2. Addition of sinusoidal positional encodings.
    3. Transformer backbone.
    4. MLP (1-layer), transforming fixed-length sequence of vectors into one vector (hack to get a single representation from a sequence).
    5. Projection layer (optional).
    6. Batch normalization (to skip that step in the contrastive loss function).

    Args:
        feature_dimension (int): Feature dimension of input data (number of frequency bins or mels).
        time_dimension (int): Time dimension of input data (number of time bins).
        model_dimension (int): Hidden size of model layers.
        projection_dimension (int): Output size of projection layer.
        layer_count (int): Number of transformer layers.
        head_count (int): Number of attention heads in each transformer layer.
    """

    def __init__(
        self,
        feature_dimension,
        time_dimension,
        model_dimension,
        projection_dimension,
        layer_count=1,
        head_count=8,
    ):
        super().__init__()

        sequence_dimension = time_dimension // 2

        self.convolution = ConvolutionEncoder(feature_dimension, model_dimension)
        positional_encoding = sinusoids(model_dimension, sequence_dimension)
        # register buffer to let the tensor "be part of the model" and transfer device together with it
        self.register_buffer("positional_encoding", positional_encoding)

        self.transformer = Transformer(model_dimension, head_count, layer_count)
        self.linear = Linear(sequence_dimension, 1, bias=False)
        self.representation_batch_norm = BatchNorm1d(model_dimension)

        self.projector = Projector(model_dimension, projection_dimension)
        self.projection_batch_norm = BatchNorm1d(projection_dimension)

    def forward(self, x, projection=True):
        """Forward pass through encoder.

        Args:
            x (torch.tensor): audio patches with shape: (batch, feature [mel, time)
            projection (bool): Whether to include the 2-layer MLP projection module in the computation.

        Returns:
           (torch.tensor): representation or projection, with shape: (batch, size) [where 'size' is 'model_dimension' if projection=False, else 'projection_dimension'].
        """
        x = self.convolution(x)

        x = x + self.positional_encoding

        # (batch, feature, sequence) ->  (sequence, batch, feature)
        x = torch.permute(x, (2, 0, 1))

        x = self.transformer(x)

        # (sequence, batch, feature) -> (batch, feature, sequence)
        x = torch.permute(x, (1, 2, 0))

        # (batch, feature, 1)
        x = relu(self.linear(x))

        # (batch, feature)
        x = torch.squeeze(x, -1)

        x = self.representation_batch_norm(x)

        if projection:
            # (batch, projection)
            x = self.projector(x)

            # TODO: check if output is normalized correctly
            x = self.projection_batch_norm(x)

        return x


class ConvolutionEncoder(Module):
    def __init__(self, input_dimension, model_dimension):
        super().__init__()

        self.convolutions = ModuleList(
            [
                Conv1d(
                    input_dimension, model_dimension, kernel_size=3, stride=1, padding=1
                ),
                Conv1d(
                    model_dimension, model_dimension, kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        self.activation = GELU()

    def forward(self, x):
        for convolution in self.convolutions:
            x = convolution(x)
            x = self.activation(x)

        return x


def sinusoids(feature_dimension, time_dimension):

    # if odd channel count, last row of positional encoding is zero
    wave_count = feature_dimension // 2
    periods = torch.arange(1, wave_count + 1)

    angles = torch.linspace(0, 1, steps=time_dimension)
    # longest wave: half period
    angles = angles.repeat(wave_count, 1) * periods.unsqueeze(1) * 3.14159
    sines = torch.sin(angles)
    cosines = torch.cos(angles)

    sinusoids = torch.zeros(feature_dimension, time_dimension)
    sinusoids[:wave_count] = sines
    sinusoids[wave_count : 2 * wave_count] = cosines

    return sinusoids


class Transformer(Module):
    def __init__(self, model_dimension, head_count=8, layer_count=1, activation="gelu"):
        super().__init__()
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=model_dimension, nhead=head_count, activation=activation
                )
                for i in range(layer_count)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Projector(Module):
    def __init__(self, model_dimension, projection_dimension):
        super().__init__()

        self.projector = Sequential(
            Linear(model_dimension, projection_dimension, bias=False),
            BatchNorm1d(projection_dimension),
            ReLU(inplace=True),
            # Linear(projection_dimension, projection_dimension, bias=False),
            # BatchNorm1d(projection_dimension),
            # ReLU(inplace=True),
            Linear(projection_dimension, projection_dimension, bias=False),
        )

    def forward(self, x):
        return self.projector(x)
