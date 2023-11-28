# import dezero

# from .dezero.layers import Deconv2d, BatchNorm
import dezero.functions as F
import dezero.layers as L
from dezero.models import Sequential, Model
import numpy as np
import dezero.core as C

use_gpu = False


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation="relu"):
    layers = []

    # Conv layer
    layers.append(
        L.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=k_size,
            stride=stride,
            pad=pad,
            nobias=True,
        )
    )

    # Batch Normalization
    if bn:
        layers.append(L.BatchNorm())

    if activation == "lrelu":
        layers.append(F.leaky_relu)
    elif activation == "relu":
        layers.append(F.relu)
    elif activation == "tanh":
        layers.append(F.tanh)
    elif activation == "none":
        # this has been added to this model for Discrimiantor
        layers.append(F.sigmoid)
        pass

    if use_gpu:
        seq = Sequential(*layers)
        seq.to_gpu()
        return seq
    else:
        return Sequential(*layers)


def deconv(
    c_in,
    c_out,
    k_size,
    stride=2,
    pad=1,
    bn=True,
    activation="lrelu",
    apply_dropout=False,
):
    layers = []

    # Deconv.
    layers.append(
        L.Deconv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=k_size,
            stride=stride,
            pad=pad,
            nobias=True,
        )
    )

    if bn:
        layers.append(L.BatchNorm())

    if activation == "lrelu":
        layers.append(F.leaky_relu)
    elif activation == "relu":
        layers.append(F.relu)
    elif activation == "tanh":
        layers.append(F.tanh)
    elif activation == "none":
        pass

    if use_gpu:
        seq = Sequential(*layers)
        seq.to_gpu()
        return seq
    else:
        return Sequential(*layers)


class Cat(C.Function):
    """
    dezeroにはcatが定義されていないので、chatgptに作ってもらった。
    """

    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *inputs):
        z = np.concatenate(inputs, axis=self.axis)
        return z

    def backward(self, gz):
        inputs = self.inputs
        gradients = []
        start_idx = 0

        for x in inputs:
            end_idx = start_idx + x.shape[self.axis]

            indices = [slice(None)] * gz.ndim
            indices[self.axis] = slice(start_idx, end_idx)

            gradients.append(gz[tuple(indices)])

            start_idx = end_idx

        return tuple(gradients)


def cat(inputs, axis=0):
    return Cat(axis=axis)(*inputs)


class Generator(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = conv(3, 64, 4, bn=False, activation="lrelu")  # (B, 64, 128, 128)
        self.conv2 = conv(64, 128, 4, activation="lrelu")  # (B, 128, 64, 64)
        self.conv3 = conv(128, 256, 4, activation="lrelu")  # (B, 256, 32, 32)
        self.conv4 = conv(256, 512, 4, activation="lrelu")  # (B, 512, 16, 16)
        self.conv5 = conv(512, 512, 4, activation="lrelu")  # (B, 512, 8, 8)
        # self.conv6 = conv(512, 512, 4, activation='lrelu') # (B, 512, 4, 4)

        # Decoder
        # self.deconv1 = deconv(512, 512, 4, activation='relu', apply_dropout=True) # (B, 512, 8, 8)
        self.deconv2 = deconv(512, 512, 4, activation="relu")  # (B, 512, 16, 16)
        self.deconv3 = deconv(
            512 * 2, 256, 4, activation="relu", apply_dropout=True
        )  # (B, 256, 32, 32) # Skip connection with concatenation or addition
        self.deconv4 = deconv(256 * 2, 128, 4, activation="relu")  # (B, 128, 64, 64)
        self.deconv5 = deconv(128 * 2, 64, 4, activation="relu")  # (B, 64, 128, 128)
        self.deconv6 = deconv(
            64 * 2, 3, 4, bn=False, activation="tanh"
        )  # (B, 3, 256, 256)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        # c6 = self.conv6(c5)

        # Decoder
        # d1 = F.dropout(self.deconv1(c6), dropout_ratio=0.5)
        d2 = F.dropout(self.deconv2(c5))
        d3 = F.dropout(self.deconv3(cat((d2, c4), axis=1)), dropout_ratio=0.5)
        d4 = self.deconv4(cat((d3, c3), axis=1))
        d5 = self.deconv5(cat((d4, c2), axis=1))
        d6 = self.deconv6(cat((d5, c1), axis=1))
        return d6
