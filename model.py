import torch.nn as nn
import torch.nn.functional as F

import configs


class encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, l2_norm=False):
        super(encoder, self).__init__()
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


class classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.network(x)
