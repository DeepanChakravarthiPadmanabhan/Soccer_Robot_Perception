import os
import timeit
import logging
import gin
import typing
import matplotlib.pyplot as plt
from sys import maxsize
import cv2
import wandb

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import time

from soccer_robot_perception.evaluate.evaluate_model import evaluate_model

from soccer_robot_perception.utils.segmentation_utils import compute_total_variation_loss_seg
from soccer_robot_perception.utils.detection_utils import compute_total_variation_loss_det


LOGGER = logging.getLogger(__name__)


@gin.configurable
class Trainer:
    """
    Trainer class. This class is used to define all the training parameters and processes for training the network.
    """

    def __init__(
        self,
        net,
        train_seg_loader,
        valid_seg_loader,
        train_det_loader,
        valid_det_loader,
        seg_criterion,
        det_criterion,
        optimizer_class,
        model_output_path,
        device,
        wandb_key,
        wandb_config,
        input_height: int,
        input_width: int,
        seg_tvl_weight: float,
        det_tvl_weight: float,
        lr_step_size=5,
        lr=1e-04,
        patience=5,
        weight_decay=0,
        num_epochs=50,
        scheduler=None,
        summary_writer=SummaryWriter(),
        evaluate=False,
        run_name="soccer-robot",
    ):
        self.net = net
        self.train_seg_loader = train_seg_loader
        self.valid_seg_loader = valid_seg_loader
        self.train_det_loader = train_det_loader
        self.valid_det_loader = valid_det_loader
        self.model_output_path = model_output_path
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.seg_criterion = seg_criterion
        self.det_criterion = det_criterion
        self.seg_tvl_weight = seg_tvl_weight
        self.det_tvl_weight = det_tvl_weight
        self.optimizer = optimizer_class([
            {"params": self.net.encoder_block1.parameters(), "lr": 0.00001},
            {"params": self.net.encoder_block2.parameters(), "lr": 0.00001},
            {"params": self.net.encoder_block3.parameters(), "lr": 0.00001},
            {"params": self.net.encoder_block4.parameters(), "lr": 0.00001},
            {"params": self.net.decoder_block1.parameters(), "lr": self.lr},
            {"params": self.net.decoder_block2.parameters(), "lr": self.lr},
            {"params": self.net.decoder_block3.parameters(), "lr": self.lr},
            {"params": self.net.segmentation_head.parameters(), "lr": self.lr},
            {"params": self.net.detection_head.parameters(), "lr": self.lr},
            {"params": self.net.conv1x1_1.parameters(), "lr": self.lr},
            {"params": self.net.conv1x1_2.parameters(), "lr": self.lr},
            {"params": self.net.conv1x1_3.parameters(), "lr": self.lr},
        ])
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=0.1)
        self.tensorboard_writer = summary_writer
        self.device = device
        self.input_height = input_height
        self.input_width = input_width
        self.loss_scale = 1.0
        self.evaluate = evaluate

        os.environ["WANDB_API_KEY"] = wandb_key
        os.environ["WANDB_NAME"] = run_name
        wandb.init(config=wandb_config, project='soccer-robot')

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
        det_train_len = len(self.train_det_loader.batch_sampler)
        seg_train_len = len(self.train_seg_loader.batch_sampler)

        LOGGER.info("Ready to start training")
        tic = timeit.default_timer()
        best_validation_loss = maxsize
        wandb.watch(self.net, log='all')

        for epoch in range(self.num_epochs):
            start = time.time()
            self.net.train(True)

            self.current_epoch = epoch + 1

            sample_size = 10

            running_loss = 0.0
            av_loss = 0.0
            running_segment_loss = 0.0
            running_regression_loss = 0.0

            for batch, det_data in enumerate(self.train_det_loader):
                if batch < seg_train_len:
                    det_data = self._sample_to_device(det_data)
                    input_image = det_data["image"]
                    self.optimizer.zero_grad()
                    det_out, seg_out = self.net(input_image)
                    det_tv_loss = compute_total_variation_loss_det(det_out, self.det_tvl_weight)
                    det_loss = det_tv_loss + self.det_criterion(det_out, det_data["det_target"])
                    seg_data = next(iter(self.train_seg_loader))
                    seg_data = self._sample_to_device(seg_data)
                    input_image = seg_data["image"]
                    det_out, seg_out = self.net(input_image)
                    seg_tv_loss = compute_total_variation_loss_seg(seg_out, self.seg_tvl_weight)
                    seg_loss = (
                            self.seg_criterion(seg_out, seg_data["seg_target"].long())
                            + seg_tv_loss
                    )
                    loss = det_loss + seg_loss
                    LOGGER.info(
                        "TRAINING - epoch: %d, step: %d, loss: %f, seg_loss: %f, det_loss: %f ",
                        self.current_epoch,
                        batch + 1,
                        loss.item(),
                        seg_loss.item(),
                        det_loss.item(),
                    )

                    loss.backward()

                    self.optimizer.step()

                    av_loss += loss.item() / self.loss_scale
                    running_loss += loss.item() / self.loss_scale
                    running_segment_loss += seg_loss.item() / self.loss_scale
                    running_regression_loss += det_loss.item() / self.loss_scale

                    if batch % sample_size == (sample_size - 1):
                        LOGGER.info(
                            "TRAINING - epoch: %d, step: %d, loss: %f, seg_loss: %f, det_loss: %f ",
                            self.current_epoch,
                            batch + 1,
                            running_loss / sample_size,
                            running_segment_loss / sample_size,
                            running_regression_loss / sample_size,
                        )
                        running_loss = 0.0

            if self.scheduler:
                self.scheduler.step()

            # output training loss
            av_train_loss = av_loss / seg_train_len

            LOGGER.info(
                "TRAIN - epoch: %d, average loss: %f",
                epoch + 1,
                av_train_loss,
            )

            av_valid_loss = self.validation()

            LOGGER.info(
                "VALIDATION - epoch: %d, average validation loss: %f",
                epoch + 1,
                av_valid_loss,
            )
            wandb.log({"train_loss": av_train_loss, "validation_loss": av_valid_loss}, step=epoch + 1)
            LOGGER.info("Current epoch completed in %f s", time.time() - start)

            if av_valid_loss < best_validation_loss and model_path and best_model_path:
            # if True:
                best_validation_loss = av_valid_loss
                best_model_path = model_path
                LOGGER.info(
                    "Better model found. Saving. Epoch %d, Path %s",
                    epoch + 1,
                    best_model_path,
                )
                torch.save(self.net.state_dict(), best_model_path)
                patience_count = self.patience
            else:
                patience_count -= 1
                LOGGER.info(
                    "No better model found. Epoch %d, Patience left %d",
                    epoch + 1,
                    patience_count,
                )

            if patience_count == 0:
                LOGGER.info(
                    "Epoch %d Patience is 0. Early stopping triggered", epoch + 1
                )
                break

        toc = timeit.default_timer()
        LOGGER.info("Finished training in %f s", toc - tic)

        if self.evaluate:
            evaluate_model(model_path=best_model_path, report_output_path="report/")

    def validation(self):
        LOGGER.info("Validation Module")
        seg_valid_len = len(self.valid_seg_loader.batch_sampler)
        det_valid_len = len(self.valid_det_loader.batch_sampler)

        self.net.train(False)

        av_loss = 0.0

        with torch.no_grad():
            for batch, det_data in enumerate(self.valid_det_loader):
                if batch < seg_valid_len:
                    det_data = self._sample_to_device(det_data)
                    input_image = det_data["image"]
                    det_out, seg_out = self.net(input_image)
                    det_tv_loss = compute_total_variation_loss_det(det_out, self.det_tvl_weight)
                    det_loss = det_tv_loss + self.det_criterion(det_out, det_data["det_target"])
                    seg_data = next(iter(self.valid_seg_loader))
                    seg_data = self._sample_to_device(seg_data)
                    input_image = seg_data["image"]
                    det_out, seg_out = self.net(input_image)
                    seg_tv_loss = compute_total_variation_loss_seg(seg_out, self.seg_tvl_weight)
                    seg_loss = (
                            self.seg_criterion(seg_out, seg_data["seg_target"].long())
                            + seg_tv_loss
                    )
                    loss = seg_loss + det_loss
                    LOGGER.info(
                        "VALIDATION - epoch: %d, step: %d, loss: %f, seg_loss: %f, det_loss: %f ",
                        self.current_epoch,
                        batch + 1,
                        loss.item(),
                        seg_loss.item(),
                        det_loss.item(),
                    )
                    av_loss += loss.item() / self.loss_scale

        av_valid_loss = av_loss / seg_valid_len
        # av_valid_loss = 0
        return av_valid_loss
