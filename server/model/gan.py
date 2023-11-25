# import dezero

# from .dezero.layers import Deconv2d, BatchNorm
import dezero.functions as F
import dezero.layers as L
from dezero.models import Sequential

Generator = Sequential(
    L.Deconv2d(in_channels=100, out_channels=512,
               kernel_size=5, stride=2, pad=0, nobias=True),
    L.BatchNorm(),
    F.relu,

    L.Deconv2d(in_channels=512, out_channels=256,
               kernel_size=4, stride=2, pad=0, nobias=True),
    L.BatchNorm(),
    F.relu,

    L.Deconv2d(in_channels=256, out_channels=128,
               kernel_size=5, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.relu,

    L.Deconv2d(in_channels=128, out_channels=3,
               kernel_size=4, stride=2, pad=1, nobias=True),
    F.tanh,
)
