import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 dropout_prob=0.5):
        """
            num_layers: number of layers in the network (EXCLUDING the input layer).
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL hidden layers
            output_dim: number of classes or output dimensions
            dropout_prob: probability of dropping units in Dropout
        """
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.latent_dim = 25
        self.dropout_prob = dropout_prob

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Simple linear model without hidden layers
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # MLP with hidden layers
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            # Input -> first hidden
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            # Hidden -> hidden
            for _ in range(num_layers - 1):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            # BatchNorm for each layer
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # Dropout applied after each activation
            self.dropout = nn.Dropout(p=self.dropout_prob)

            # 瓶颈层之后的 BatchNorm
            self.bn_latent = nn.BatchNorm1d(self.latent_dim)

            # Final projection through a latent bottleneck
            self.fc1 = nn.Linear(hidden_dim, self.latent_dim)
            self.fc2 = nn.Linear(self.latent_dim, output_dim)

    def forward(self, x):
        # Ensure float32 for batchnorm
        x = x.float()

        if self.num_layers == 1:
            # Linear model: return raw output (logits or unbounded values)
            return self.linear(x)
        else:
            h = x
            # Hidden layers: Linear -> BatchNorm -> ReLU -> Dropout
            for i in range(self.num_layers):
                h = self.linears[i](h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)

            # Bottleneck projection
            out = self.fc1(h)
            # out = self.bn_latent(out) #
            out = F.relu(out)
            # out = self.dropout(out) #
            out = self.fc2(out)
            # out = torch.flatten(torch.sigmoid(out))
            out = torch.flatten(out)

            # Return raw outputs (logits) without sigmoid
            return out
