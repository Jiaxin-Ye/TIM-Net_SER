import torch
import torch.nn.functional as F


class LabelSmoothingLoss(torch.nn.Module):
    # Source: https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631

    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, ::], self.y[idx]


class CausalConv1d(torch.nn.Conv1d):
    # Source: https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class Temporal_Aware_Block(torch.nn.Module):
    def __init__(self, i, nb_filters, kernel_size, dropout):
        super().__init__()

        self.i = i
        self.dropout = dropout
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size

        self.conv1d_1 = CausalConv1d(
            in_channels=self.nb_filters,
            out_channels=self.nb_filters,
            kernel_size=self.kernel_size,
            dilation=i,
        )
        self.conv1d_2 = CausalConv1d(
            in_channels=self.nb_filters,
            out_channels=self.nb_filters,
            kernel_size=self.kernel_size,
            dilation=i,
        )

        self.batchnorm_1 = torch.nn.BatchNorm1d(self.nb_filters)
        self.batchnorm_2 = torch.nn.BatchNorm1d(self.nb_filters)

    def forward(self, input):
        original_input = input

        conv_1_1 = self.conv1d_1(input)
        conv_1_1 = self.batchnorm_1(conv_1_1)
        conv_1_1 = F.relu(conv_1_1)
        output_1_1 = F.dropout1d(conv_1_1, p=self.dropout, training=self.training)

        conv_2_1 = self.conv1d_2(output_1_1)
        conv_2_1 = self.batchnorm_2(conv_2_1)
        conv_2_1 = F.relu(conv_2_1)
        output_2_1 = F.dropout1d(conv_2_1, p=self.dropout, training=self.training)

        output_2_1 = torch.sigmoid(output_2_1)
        out = torch.mul(original_input, output_2_1)

        return out


class TorchTIMNET(torch.nn.Module):
    def __init__(
        self,
        dropout,
        nb_filters,
        kernel_size,
        dilations,
        output_dim,
    ):
        super().__init__()

        self.dropout = dropout
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.output_dim = output_dim

        self.forward_conv1d = CausalConv1d(
            in_channels=self.nb_filters,
            out_channels=self.nb_filters,
            kernel_size=1,
            dilation=1,
        )
        self.backward_conv1d = CausalConv1d(
            in_channels=self.nb_filters,
            out_channels=self.nb_filters,
            kernel_size=1,
            dilation=1,
        )
        self.dilation_values = [2 ** i for i in range(self.dilations)]

        self.forward_temporal = torch.nn.ModuleList(
            [
                Temporal_Aware_Block(i, self.nb_filters, self.kernel_size, self.dropout)
                for i in self.dilation_values
            ]
        )
        self.backward_temporal = torch.nn.ModuleList(
            [
                Temporal_Aware_Block(i, self.nb_filters, self.kernel_size, self.dropout)
                for i in self.dilation_values
            ]
        )
        self.weight = torch.nn.Linear(len(self.dilation_values), 1, bias=False)
        self.weight2 = torch.nn.Linear(self.nb_filters, self.output_dim)


    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        forward_x = x
        backward_x = torch.flip(x, dims=(2,))

        skip_forward = self.forward_conv1d(forward_x)
        skip_backward = self.backward_conv1d(backward_x)

        skip_connections = []

        for i in range(len(self.forward_temporal)):
            skip_forward = self.forward_temporal[i](skip_forward)
            skip_backward = self.backward_temporal[i](skip_backward)

            temp = skip_forward + skip_backward
            temp = F.adaptive_avg_pool1d(temp, 1)

            skip_connections.append(temp)

        ct = torch.cat(skip_connections, dim=-1)
        ct = self.weight(ct).squeeze()
        ct = self.weight2(ct)

        return ct
