import os
import timeit
import logging
import gin
import typing
import matplotlib.pyplot as plt
from sys import maxsize

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import time

LOGGER = logging.getLogger(__name__)


@gin.configurable
class Trainer:
    """
    Trainer class. This class is used to define all the training parameters and processes for training the network.
    """

    def __init__(self):
        pass
