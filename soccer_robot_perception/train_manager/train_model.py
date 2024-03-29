import logging
import typing
import gin
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.data import Dataset
from soccer_robot_perception.trainer.trainer import Trainer

LOGGER = logging.getLogger(__name__)


@gin.configurable
def train_model(
    tensorboard_dir: str,
    model_output_dir: str,
    input_height: int,
    input_width: int,
    net: torch.nn.Module,
    seg_data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
    det_data_loaders: typing.Tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
        ],
):
    # load cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device used for training: %s", device.type)

    # tensorboard writer
    tensorboard_writer = (
        SummaryWriter(tensorboard_dir) if tensorboard_dir else SummaryWriter()
    )

    train_seg_loader, valid_seg_loader, test_seg_loader = seg_data_loaders
    train_det_loader, valid_det_loader, test_det_loader = det_data_loaders

    net.to(device)
    summary(
        net,
        input_size=(3, input_height, input_width),
        batch_size=train_seg_loader.batch_size,
        device=device.type,
    )
    trainer = Trainer(
        net=net,
        train_seg_loader=train_seg_loader,
        valid_seg_loader=valid_seg_loader,
        train_det_loader=train_det_loader,
        valid_det_loader=valid_det_loader,
        device=device,
        summary_writer=tensorboard_writer,
        model_output_path=model_output_dir,
        input_width=input_width,
        input_height=input_height,
    )
    trainer.train()

    # # Task completion information
    LOGGER.info("Train script completed")
