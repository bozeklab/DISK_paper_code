"""Define mbi models."""
# import keras
# from keras import layers
# from keras.layers import Input, Conv1d, Dense, Lambda, Permute, LSTM
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.regularizers import l2
import torch
from torch.nn import Conv1d, Linear

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
        # # Specify the Input
        # history_seq = Input(shape=(input_length, n_markers))
        # x = history_seq

        # Dilated causal convolutions
        dilation_rates = [2**i for i in range(n_dilations)]
        self.convs = []
        n_in = n_markers
        self.input_length = input_length
        out_len_because_padding = int(input_length)
        for dilation_rate in dilation_rates:
            for i in range(layers_per_level):
                self.convs.append(Conv1d(in_channels=n_in,
                                         out_channels=n_filters,
                                         kernel_size=filter_width,
                                         padding=dilation_rate * (filter_width - 1),
                                         dilation=dilation_rate).to(device))
                # torch.nn.init.xavier_uniform()
                # torch.nn.init.kaiming_normal_(self.convs[-1].weight, mode='fan_in', nonlinearity='relu')
                #out_len_because_padding += dilation_rate * (filter_width - 1)
                # filters=n_filters,
                #                            kernel_size=filter_width,
                #                            padding='causal',
                #                            dilation_rate=dilation_rate,
                #                            activation='relu'))
                n_in = n_filters

        self.downsample = Conv1d(n_markers, n_filters, 1) if n_markers != n_filters else None

        # Dense connections
        self.linear_on_marker_dim = Linear(n_filters, n_markers)
        self.linear_on_time_dimension = Linear(self.input_length, output_length)

        # Build and compile the model
        # model = Model(history_seq, x)
        # # model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])
        # if print_summary:
        #     model.summary()

    def forward(self, x):
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

#
# def wave_net_res_skip(lossfunc, lr, input_length, n_filters, n_markers,
#                       n_dilations, layers_per_level, filter_width,
#                       use_skip_connections=False, learn_all_outputs=False,
#                       use_bias=True, res_l2=.01, print_summary=False):
#     """Wave_net model with residual and optional skip skip_connections.
#
#     :param lossfunc: Loss function
#     :param lr: Loss rate
#     :param input_length: Model input length (frames)
#     :param n_filters: Number of filters per convolutional block
#     :param n_markers: Number of markers per frame
#     :param n_dilations: Number of dilation levels
#     :param layers_per_level: Number of convolutional blocks per dilation level
#     :param filter_width: Width of convolutional filters (frames)
#     :param use_skip_connections: Whether or not to use skip connections
#     :param use_bias: Whether to use bias in residual convolutions
#     :param res_l2: Residual skip connection regularizer
#     :param print_summary: Print the model specification.
#     """
#     # TODO(Output): include output_length specification.
#     def residual_block(x):
#         original_x = x
#         x = Conv1d(filters=n_filters,
#                    kernel_size=filter_width,
#                    padding='causal',
#                    name='dilated_conv_%d_s%d_01' % (2 ** i, s),
#                    dilation_rate=2**i)(x)
#         res_x = Conv1d(filters=n_filters,
#                        kernel_size=filter_width,
#                        padding='causal',
#                        name='dilated_conv_%d_s%d_02' % (2 ** i, s),
#                        dilation_rate=2**i)(x)
#         res_x = Conv1d(filters=n_filters,
#                        kernel_size=filter_width,
#                        padding='causal',
#                        name='dilated_conv_%d_s%d_03' % (2 ** i, s),
#                        dilation_rate=2**i)(res_x)
#         skip_x = Conv1d(n_filters, 1, padding='same', use_bias=use_bias,
#                         kernel_regularizer=l2(res_l2))(x)
#         res_x = layers.Add()([original_x, res_x])
#         return res_x, skip_x
#
#     # Set the dilations
#     input = Input(shape=(input_length, n_markers), name='input_part')
#     out = input
#     skip_connections = []
#     out = Conv1d(n_filters, filter_width, dilation_rate=1, padding='causal',
#                  name='initial_causal_conv')(out)
#
#     # Construct the residual blocks
#     for i in range(0, n_dilations + 1):
#         for s in range(layers_per_level):
#             out, skip_out = residual_block(out)
#             skip_connections.append(skip_out)
#
#     # Optional skip connections
#     if use_skip_connections:
#         out = layers.Add()(skip_connections)
#
#     # Dense connections
#     out = Dense(n_markers)(out)
#     out = torch.permute([2, 1])(out)
#     out = Dense(1)(out)
#     out = torch.permute([2, 1])(out)
#
#     # Build and compile the model
#     model = Model(input, out)
#     model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])
#     if print_summary:
#         model.summary()
#     return model
#
#
# def lstm_model(lossfunc, lr, input_length, n_markers, latent_dim,
#                print_summary=False):
#     """Lstm base model.
#
#     :param lossfunc: Loss function
#     :param lr: Loss rate
#     :param input_length: Model input length (frames)
#     :param n_markers: Number of markers per frame
#     :param latent_dim: Number of latent dimensions in LSTM layers
#     :param print_summary: Print the model specification.
#     """
#     inputs1 = Input(shape=(input_length, n_markers))
#     encoded1 = LSTM(latent_dim, return_sequences=True)(inputs1)
#     encoded1 = LSTM(latent_dim, return_sequences=True)(encoded1)
#     encoded1 = LSTM(latent_dim, return_sequences=False)(encoded1)
#     encoded = Linear(n_markers)(encoded1)
#
#     # Include this to be consistent with [nSample, nFrame, nMarker]
#     # output format
#     def pad(x):
#         return(x[:, None, :])
#     encoded = Lambda(pad)(encoded)
#
#     # Build and compile model
#     model = Model(inputs1, encoded)
#     model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])
#
#     if print_summary:
#         model.summary()
#
#     return model
#
#
# def conv_lstm(lossfunc, lr, input_length, output_length, n_markers,
#               n_filters, filter_width, layers_per_level, n_dilations,
#               latent_dim, print_summary=False):
#     """Convolutional LSTM model.
#
#     Not thoroughly tested.
#     Not implemented in training.py.
#     :param lossfunc: Loss function
#     :param lr: Loss rate
#     :param input_length: Model input length (frames)
#     :param output_length: Model output length (frames)
#     :param n_markers: Number of markers per frame
#     :param n_filters: Number of filters per convolutional block
#     :param filter_width: Width of convolutional filters (frames)
#     :param layers_per_level: Number of convolutional blocks per dilation level
#     :param n_dilations: Number of dilation levels
#     :param latent_dim: Number of latent dimensions in LSTM layers
#     :param print_summary: Print the model specification.
#     """
#     # Specify the Input
#     history_seq = Input(shape=(input_length, n_markers))
#     x = history_seq
#
#     # Dilated causal convolutions
#     dilation_rates = [2**i for i in range(n_dilations)]
#     for dilation_rate in dilation_rates:
#         for i in range(layers_per_level):
#             x = Conv1d(filters=n_filters,
#                        kernel_size=filter_width,
#                        padding='causal',
#                        dilation_rate=dilation_rate)(x)
#
#     encoded1 = LSTM(latent_dim, return_sequences=False)(x)
#
#     # Linear connections
#     x = Linear(60)(x)
#     x = torch.permute(x, [2, 1])
#     x = Linear(output_length)(x)
#     x = torch.permute(x, [2, 1])
#
#     model = Model(history_seq, x)
#     model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])
#     if print_summary:
#         model.summary()
#
#     return model