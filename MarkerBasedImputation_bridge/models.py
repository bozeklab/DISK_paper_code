"""Define mbi models."""
# import keras
# from keras import layers
# from keras.layers import Input, Conv1d, Dense, Lambda, Permute, LSTM
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.regularizers import l2
import numpy as np
import torch
from torch.nn import Conv1d, Linear, ModuleList


class Wave_net(torch.nn.Module):
    def __init__(self, input_length, output_length, n_markers, n_filters,
             filter_width, layers_per_level, n_dilations, device,
             print_summary=False, **kwargs):
        """Build the base WaveNet model.

        :param lossfunc: Loss function
        :param lr: Loss rate
        :param input_length: Model input length (frames)
        :param output_length: Model output length (frames)
        :param n_markers: Number of markers per frame
        :param n_filters: Number of filters per convolutional block
        :param filter_width: Width of convolutional filters (frames)
        :param layers_per_level: Number of convolutional blocks per dilation level
        :param n_dilations: Number of dilation levels
        :param print_summary: Print the model specification.
        """
        super(Wave_net, self).__init__()

        self.input_length = input_length
        self.output_length = output_length

        # Dilated causal convolutions
        dilation_rates = [2**i for i in range(n_dilations)]
        n_in = n_markers
        self.convs = ModuleList()
        for dilation_rate in dilation_rates:
            for i in range(layers_per_level):
                self.convs.append(Conv1d(in_channels=n_in,
                                         out_channels=n_filters,
                                         kernel_size=filter_width,
                                         padding=dilation_rate * (filter_width - 1),
                                         dilation=dilation_rate).to(device))
                # from keras: filters=n_filters, kernel_size=filter_width, padding='causal',
                # dilation_rate=dilation_rate, activation='relu'
                n_in = n_filters

        self.downsample = Conv1d(n_markers, n_filters, 1) if n_markers != n_filters else None

        # Dense connections
        self.linear_on_marker_dim = Linear(n_filters, n_markers)
        self.linear_on_time_dimension = Linear(self.input_length, self.output_length)


    def forward(self, x, verbose=False):
        x_ = torch.swapaxes(x, 1, 2) # shapes: (batch, len, markers) -> (batch, markers, len)

        for i_c, c in enumerate(self.convs):
            if i_c == 0:
                res = x_ if self.downsample is None else self.downsample(x_)
                # x_ = torch.relu(c(x_)[..., :self.input_length].contiguous()) # after convs, shape (batch, filters, len + padding)
            else:
                res = x_
            x_ = torch.relu(c(x_)[..., :self.input_length].contiguous() + res)  # after convs, shape (batch, filters, len + padding)

        x_ = torch.permute(x_, [0, 2, 1]) # shape: (batch, filters, len + padding) -> (batch, len + padding, filters)
        x_ = self.linear_on_marker_dim(x_) # shape: (batch, len + padding, markers)
        x_ = torch.permute(x_, [0, 2, 1]) # shape: (batch, len + padding, markers) -> (batch, markers, len + padding)
        x_ = self.linear_on_time_dimension(x_) # shape: (batch, markers, output_len = 1)
        x_ = torch.permute(x_, [0, 2, 1]) # shape: (batch, markers, output_len = 1) -> (batch, output_len = 1, markers)

        return x_
