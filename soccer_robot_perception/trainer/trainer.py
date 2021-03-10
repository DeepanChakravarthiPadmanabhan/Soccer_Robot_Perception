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

    def __init__(self,
                 net,
                 train_loader,
                 valid_loader,
                 seg_criterion,
                 det_criterion,
                 optimizer_class,
                 model_output_path,
                 device,
                 input_height: int,
                 input_width: int,
                 lr_step_size=5,
                 lr=1e-04,
                 patience=5,
                 weight_decay=0,
                 num_epochs=50,
                 scheduler=None,
                 summary_writer=SummaryWriter(),
                 ):
        self.net = net
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_output_path = model_output_path
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.seg_criterion = seg_criterion
        self.det_criterion = det_criterion
        self.optimizer = optimizer_class(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=0.1)
        self.tensorboard_writer = summary_writer
        self.device = device
        self.input_height = input_height
        self.input_width = input_width

    def _sample_to_device(self, sample):
        device_sample = {}
        for key, value in sample.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    device_sample[key] = value.to(self.device)
                elif isinstance(value, list):
                    device_sample[key] = [
                        t.to(self.device) if isinstance(t, torch.Tensor) else t
                        for t in value
                    ]
                else:
                    device_sample[key] = value

        return device_sample

    def train(self):

        LOGGER.info("Train Module")
        if not os.path.exists(os.path.dirname(self.model_output_path)):
            LOGGER.info(
                "Output directory does not exist. Creating directory %s",
                os.path.dirname(self.model_output_path),
            )
            os.makedirs(os.path.dirname(self.model_output_path))
        model_path = os.path.join(self.model_output_path, "model.pth")
        best_model_path = model_path
        patience_count = self.patience
        train_len = len(self.train_loader.batch_sampler)

        LOGGER.info("Ready to start training")
        tic = timeit.default_timer()
        best_validation_loss = maxsize

        for epoch in range(self.num_epochs):
            start = time.time()
            self.current_epoch = epoch + 1
            sample_size = 10
            self.net.train(True)

            running_loss = 0.0
            av_loss = 0.0
            running_segment_loss = 0.0
            running_regression_loss = 0.0

            for batch, data in enumerate(self.train_loader):
                LOGGER.info("Training: batch %d of epoch %d", batch + 1, epoch + 1)
                data = self._sample_to_device(data)

                input_image = data["image"]
                self.optimizer.zero_grad()

                det_out, seg_out = self.net(input_image)
                print('Output shapes: ', det_out.shape, seg_out.shape)

                # To calculate loss for each data
                # for n,i in enumerate(data["dataset_class"]):
                #     if i == 'detection':
                #         det_target = data["target"][n]
                #         # loss =
                #     else:
                #         seg_target = data["target"][n]
                #         loss = self.seg_criterion(seg_out, seg_target)
                #         print(loss)


            break
