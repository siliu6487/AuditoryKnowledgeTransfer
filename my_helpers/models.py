import torch.nn as nn
import torch.nn.functional as F
import torch


class GeneralEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, l2_norm):
        super(GeneralEncoder, self).__init__()
        super().__init__()
        self.l2_norm = l2_norm
        self.output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        output = self.network(x)
        if self.l2_norm:
            output = F.normalize(output, p=2, dim=-1)
        return output


class LinearProbLayer(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        """
        A simple linear probe layer for classification.

        Args:
            in_dim (int): Dimension of the encoder's output features.
            num_classes (int): Number of classes for classification.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

